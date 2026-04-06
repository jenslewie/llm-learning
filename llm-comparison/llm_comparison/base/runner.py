import argparse
from datetime import datetime
import time
from pathlib import Path

from llm_comparison.base.prompts import build_prompt, extract_final_answer_json, get_question_text, get_display_gold, load_questions
from llm_comparison.base.reporting import build_compare_table, build_summary, save_run_artifacts
from llm_comparison.base.scoring import add_scores
from llm_comparison.common.console import configure_stdout, format_summary_table, log
from llm_comparison.common.files import resolve_cli_path, write_json, load_json
from llm_comparison.common.models import load_models_config, select_models
from llm_comparison.common.runtime import (
    build_generation_prompt,
    extract_stream_metrics,
    get_backend_bundle,
    release_model_resources,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local LLMs one by one.")
    parser.add_argument("--config", default="config/models_config.json", help="Path to the models config JSON file.")
    parser.add_argument("--models", nargs="*", help="Optional model ids to run. Defaults to all models in the config.")
    parser.add_argument("--questions", help="Optional question set JSON file. Defaults to the built-in 12 questions.")
    parser.add_argument("--output-root", default="benchmark_results", help="Directory where run artifacts are stored.")
    parser.add_argument("--run-name", help="Optional run name. Defaults to a timestamp like 20260404_153000.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-model result JSON files inside the run directory.")
    return parser.parse_args()


def run_one_model(model_config: dict, tests: list[dict]) -> list[dict]:
    backend = model_config["backend"]
    backend_bundle = get_backend_bundle(backend)
    enable_thinking = bool(model_config.get("enable_thinking", False))
    max_tokens = int(model_config.get("max_tokens", 512))
    model_name = model_config["model_name"]
    model_id = model_config["id"]

    model = None
    processor = None
    config = None
    log(f"Preparing model: {model_id}")
    log(f"Model path: {model_name}")
    log(f"Backend: {backend}")
    log(f"enable_thinking={enable_thinking}")
    log(f"[{model_id}] Starting model load")
    model, processor = backend_bundle["load"](model_name)
    log(f"[{model_id}] Model load complete")
    if backend == "mlx_vlm":
        log(f"[{model_id}] Loading backend config")
        config = backend_bundle["load_config"](model_name)
        log(f"[{model_id}] Backend config loaded")
    log(f"[{model_id}] Model ready")

    results = []
    try:
        for index, item in enumerate(tests, 1):
            prompt_text = build_prompt(get_question_text(item))
            messages = [{"role": "user", "content": prompt_text}]
            log(f"[{model_id}] [{index}/{len(tests)}] Building prompt for {item['id']}")
            prompt = build_generation_prompt(
                backend=backend,
                backend_bundle=backend_bundle,
                processor=processor,
                config=config,
                messages=messages,
                enable_thinking=enable_thinking,
            )

            log(f"[{model_id}] [{index}/{len(tests)}] Starting generation for {item['id']}")
            started_at = time.time()
            stream_kwargs = {"prompt": prompt, "max_tokens": max_tokens}
            if backend == "mlx_vlm":
                stream_kwargs["enable_thinking"] = enable_thinking

            text_parts = []
            last_response = None
            for response in backend_bundle["stream_generate"](model, processor, **stream_kwargs):
                text_parts.append(response.text)
                last_response = response
            latency = time.time() - started_at
            log(f"[{model_id}] [{index}/{len(tests)}] Generation finished for {item['id']} in {latency:.2f}s")

            text = "".join(text_parts).strip()
            metrics = extract_stream_metrics(last_response) if last_response is not None else {
                "prompt_tokens": None,
                "generation_tokens": None,
                "prompt_tps": None,
                "generation_tps": None,
                "prefill_sec": None,
                "decode_sec": None,
                "peak_memory_gb": None,
            }
            final_pred = extract_final_answer_json(text)

            results.append(
                {
                    "model_id": model_id,
                    "model_label": model_config.get("label", model_id),
                    "model": model_name,
                    "backend": backend,
                    "enable_thinking": enable_thinking,
                    "qid": item["id"],
                    "question_category": item.get("category"),
                    "evaluation_type": item.get("evaluation_type", "objective"),
                    "scoring_rule": item.get("scoring_rule"),
                    "question": get_question_text(item),
                    "gold": get_display_gold(item),
                    "pred": text,
                    "final_pred": final_pred,
                    "latency_sec": round(latency, 2),
                    "pred_len": len(text),
                    "final_pred_len": len(final_pred),
                    **metrics,
                }
            )
    finally:
        release_model_resources(model=model, processor=processor)
        log(f"Model released: {model_id}")

    return results


def main():
    configure_stdout()
    args = parse_args()
    script_dir = Path(__file__).resolve().parents[2]
    config_path = resolve_cli_path(script_dir, args.config)
    questions_path = resolve_cli_path(script_dir, args.questions)
    output_root = resolve_cli_path(script_dir, args.output_root)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_name

    models = load_models_config(config_path)
    question_set, tests = load_questions(questions_path)
    questions_by_id = {question["id"]: question for question in tests}
    selected_models = select_models(models, args.models)

    log(f"Using config: {config_path}")
    if questions_path is not None:
        log(f"Using questions: {questions_path}")
        log(f"Question set: {question_set.get('name', 'unknown')} [version={question_set.get('version', 'unknown')}, size={len(tests)}]")
    else:
        log(f"Using built-in questions: {len(tests)}")
    log(f"Run output directory: {output_dir}")
    log("Selected models:")
    for model in selected_models:
        log(f"- {model['id']}: {model['model_name']} [backend={model['backend']}, enable_thinking={bool(model.get('enable_thinking', False))}]")

    model_results = []
    for model_config in selected_models:
        result_path = output_dir / f"results_{model_config['id']}.json"
        if args.skip_existing and result_path.exists():
            log(f"Skipping existing result: {result_path}")
            results = load_json(result_path)
        else:
            log(f"Starting benchmark for model {model_config['id']}")
            results = run_one_model(model_config, tests)
            write_json(result_path, results)
            log(f"Saved result: {result_path}")
        model_results.append(add_scores(results, questions_by_id))

    summary = build_summary(model_results)
    compare = build_compare_table(model_results, tests)
    save_run_artifacts(output_dir, config_path, questions_path, question_set, selected_models, summary, compare)

    log("Summary:")
    print(format_summary_table(summary), flush=True)
    log(f"Saved summary: {output_dir / 'summary.csv'}")
    log(f"Saved compare: {output_dir / 'compare.csv'}")
