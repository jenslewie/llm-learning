import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from llm_comparison.common.console import configure_stdout, format_summary_table, log
from llm_comparison.common.files import load_json, resolve_cli_path, write_json
from llm_comparison.common.models import load_models_config, select_models
from llm_comparison.common.runtime import (
    build_generation_prompt,
    extract_stream_metrics,
    get_backend_bundle,
    release_model_resources,
)
from llm_comparison.tooling.prompts import build_tool_prompt, load_tool_questions, maybe_inject_first_call_argument_error, parse_action_json
from llm_comparison.tooling.reporting import build_compare_table, build_summary, save_run_artifacts
from llm_comparison.tooling.runtime import run_tool_and_trace
from llm_comparison.tooling.scoring import score_tool_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tool-use benchmark with mock tools.")
    parser.add_argument("--config", default="config/models_config.json", help="Path to the models config JSON file.")
    parser.add_argument("--questions", default="benchmarks/questions/tool_questions.json", help="Path to the tool question-set JSON file.")
    parser.add_argument("--models", nargs="*", help="Optional model ids to run. Defaults to all models in the config.")
    parser.add_argument("--output-root", default="benchmark_results", help="Directory where run artifacts are stored.")
    parser.add_argument("--run-name", help="Optional run name. Defaults to a timestamp like 20260406_180000.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-model result JSON files inside the run directory.")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum assistant-tool turns per question.")
    return parser.parse_args()


def run_one_model(model_config: dict, questions: list[dict], max_steps: int) -> list[dict]:
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
        for index, question in enumerate(questions, 1):
            messages = [{"role": "user", "content": build_tool_prompt(question)}]
            traces = []
            total_latency = 0.0
            total_prompt_tokens = 0
            total_generation_tokens = 0
            total_prefill_sec = 0.0
            total_decode_sec = 0.0
            peak_memory_gb = None
            raw_steps = []
            final_pred = ""

            log(f"[{model_id}] [{index}/{len(questions)}] Starting tool task {question['id']}")
            for step in range(1, max_steps + 1):
                prompt = build_generation_prompt(
                    backend=backend,
                    backend_bundle=backend_bundle,
                    processor=processor,
                    config=config,
                    messages=messages,
                    enable_thinking=enable_thinking,
                )
                stream_kwargs = {"prompt": prompt, "max_tokens": max_tokens}
                if backend == "mlx_vlm":
                    stream_kwargs["enable_thinking"] = enable_thinking

                started_at = time.time()
                text_parts = []
                last_response = None
                for response in backend_bundle["stream_generate"](model, processor, **stream_kwargs):
                    text_parts.append(response.text)
                    last_response = response
                latency = time.time() - started_at
                total_latency += latency

                step_text = "".join(text_parts).strip()
                action = parse_action_json(step_text)
                metrics = extract_stream_metrics(last_response) if last_response is not None else {}
                total_prompt_tokens += metrics.get("prompt_tokens") or 0
                total_generation_tokens += metrics.get("generation_tokens") or 0
                total_prefill_sec += metrics.get("prefill_sec") or 0
                total_decode_sec += metrics.get("decode_sec") or 0
                if metrics.get("peak_memory_gb") is not None:
                    peak_memory_gb = max(peak_memory_gb or 0, metrics["peak_memory_gb"])

                raw_steps.append({"step": step, "assistant_text": step_text, "action": action, "latency_sec": round(latency, 2), **metrics})
                messages.append({"role": "assistant", "content": step_text})

                if action.get("action") == "tool":
                    tool_name = action.get("tool_name", "")
                    model_arguments = action.get("arguments", {})
                    executed_arguments, injection_meta = maybe_inject_first_call_argument_error(question, tool_name, model_arguments, traces)
                    if tool_name not in question.get("allowed_tools", []):
                        output = None
                        trace = {
                            "tool_name": tool_name,
                            "arguments": executed_arguments,
                            "call_started_at": time.time(),
                            "call_finished_at": time.time(),
                            "latency_sec": 0.0,
                            "success": False,
                            "error_message": "Tool not allowed for this question",
                            "output_excerpt": "",
                        }
                    else:
                        output, trace = run_tool_and_trace(tool_name, executed_arguments)
                    if injection_meta is not None:
                        trace["injection_meta"] = injection_meta
                    traces.append(trace)
                    tool_feedback = {
                        "tool_name": tool_name,
                        "success": trace["success"],
                        "error_message": trace["error_message"],
                        "output": output,
                    }
                    messages.append({"role": "user", "content": "工具调用结果如下，请继续只输出一个 JSON 对象：\n" + json.dumps(tool_feedback, ensure_ascii=False)})
                    continue

                final_answer = action.get("final_answer", "")
                final_pred = json.dumps(final_answer, ensure_ascii=False) if isinstance(final_answer, (dict, list)) else str(final_answer).strip()
                break

            scoring = score_tool_question(question, {"final_pred": final_pred, "tool_trace": traces})
            results.append(
                {
                    "model_id": model_id,
                    "model_label": model_config.get("label", model_id),
                    "model": model_name,
                    "backend": backend,
                    "enable_thinking": enable_thinking,
                    "qid": question["id"],
                    "category": question.get("category"),
                    "task": question["task"],
                    "allowed_tools": question.get("allowed_tools", []),
                    "expected_tool_sequence": question.get("expected_tool_sequence", []),
                    "gold": question.get("gold"),
                    "gold_keywords": question.get("gold_keywords"),
                    "gold_json": question.get("gold_json"),
                    "final_pred": final_pred,
                    "tool_trace": traces,
                    "raw_steps": raw_steps,
                    "latency_sec": round(total_latency, 2),
                    "tool_call_count": len(traces),
                    "successful_tool_calls": sum(1 for trace in traces if trace["success"]),
                    "prompt_tokens": total_prompt_tokens or None,
                    "generation_tokens": total_generation_tokens or None,
                    "prefill_sec": round(total_prefill_sec, 4) if total_prefill_sec else None,
                    "decode_sec": round(total_decode_sec, 4) if total_decode_sec else None,
                    "prompt_tps": round(total_prompt_tokens / total_prefill_sec, 4) if total_prefill_sec else None,
                    "generation_tps": round(total_generation_tokens / total_decode_sec, 4) if total_decode_sec else None,
                    "peak_memory_gb": peak_memory_gb,
                    **scoring,
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
    selected_models = select_models(models, args.models)
    question_set, questions = load_tool_questions(questions_path)

    log(f"Using config: {config_path}")
    log(f"Using tool questions: {questions_path}")
    log(f"Question set: {question_set.get('name', 'unknown')} [version={question_set.get('version', 'unknown')}, size={len(questions)}]")
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
            log(f"Starting tool benchmark for model {model_config['id']}")
            results = run_one_model(model_config, questions, args.max_steps)
            write_json(result_path, results)
            log(f"Saved result: {result_path}")
        model_results.append(results)

    summary = build_summary(model_results)
    compare = build_compare_table(model_results, questions)
    save_run_artifacts(output_dir, config_path, questions_path, selected_models, summary, compare)

    log("Summary:")
    print(format_summary_table(summary), flush=True)
    log(f"Saved summary: {output_dir / 'summary.csv'}")
    log(f"Saved compare: {output_dir / 'compare.csv'}")
