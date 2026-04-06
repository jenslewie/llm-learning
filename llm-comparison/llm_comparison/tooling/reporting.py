from datetime import datetime
from pathlib import Path

from llm_comparison.common.files import write_csv, write_json
from llm_comparison.common.stats import safe_average, safe_ratio


def build_summary(model_results: list[list[dict]]) -> list[dict]:
    rows = []
    for results in model_results:
        first = results[0]
        tool_calls = [row["tool_call_count"] for row in results]
        total_tool_calls = sum(tool_calls)
        successful_tool_calls = sum(row["successful_tool_calls"] for row in results)
        recovery_rows = [row for row in results if row["qid"] == "tq06"]
        recovery_success = sum(
            1
            for row in recovery_rows
            if any(not trace["success"] for trace in row["tool_trace"]) and any(trace["success"] for trace in row["tool_trace"][1:])
        )
        total_prompt_tokens = sum(row["prompt_tokens"] or 0 for row in results)
        total_prefill_sec = sum(row["prefill_sec"] or 0 for row in results)

        rows.append(
            {
                "model_id": first["model_id"],
                "model_label": first["model_label"],
                "model": first["model"],
                "backend": first["backend"],
                "enable_thinking": bool(first["enable_thinking"]),
                "total_score": int(sum(row["score"] for row in results)),
                "total_max_score": int(sum(row["max_score"] for row in results)),
                "score_rate": safe_ratio(sum(row["score"] for row in results), sum(row["max_score"] for row in results)),
                "full_score_count": int(sum(row["full_score"] for row in results)),
                "avg_latency_sec": round(sum(row["latency_sec"] for row in results) / len(results), 2),
                "avg_tool_calls_per_task": safe_average(tool_calls),
                "tool_call_success_rate": safe_ratio(successful_tool_calls, total_tool_calls),
                "end_to_end_success_rate": safe_ratio(sum(row["end_to_end_success"] for row in results), len(results)),
                "recovery_rate": safe_ratio(recovery_success, len(recovery_rows)) if recovery_rows else None,
                "avg_prefill_sec": safe_average([row["prefill_sec"] for row in results]),
                "avg_generation_tps": safe_average([row["generation_tps"] for row in results]),
                "weighted_prompt_tps": safe_ratio(total_prompt_tokens, total_prefill_sec),
            }
        )
    return sorted(rows, key=lambda row: (-(row["score_rate"] or 0), -row["full_score_count"], row["avg_latency_sec"]))


def build_compare_table(model_results: list[list[dict]], questions: list[dict]) -> list[dict]:
    compare = [{"qid": question["id"], "category": question.get("category"), "task": question.get("task")} for question in questions]
    by_qid = {row["qid"]: row for row in compare}
    for results in model_results:
        model_id = results[0]["model_id"]
        for row in results:
            target = by_qid[row["qid"]]
            target[f"final_pred__{model_id}"] = row["final_pred"]
            target[f"score__{model_id}"] = row["score"]
            target[f"tool_calls__{model_id}"] = row["tool_call_count"]
    return compare


def save_run_artifacts(output_dir: Path, config_path: Path, questions_path: Path, selected_models: list[dict], summary: list[dict], compare: list[dict]):
    snapshot = {
        "config_path": str(config_path),
        "questions_path": str(questions_path),
        "selected_model_ids": [model["id"] for model in selected_models],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(output_dir / "run_config_snapshot.json", snapshot)
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "compare.csv", compare)
    write_json(output_dir / "compare.json", compare)

