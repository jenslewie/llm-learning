from datetime import datetime
from pathlib import Path

from llm_comparison.base.prompts import get_display_gold
from llm_comparison.common.files import write_csv, write_json
from llm_comparison.common.stats import safe_average, safe_ratio


def build_summary(model_results: list[list[dict]]) -> list[dict]:
    records = []
    for rows in model_results:
        first = rows[0]
        cold_start = rows[0]
        steady_state_rows = rows[1:]
        total_prompt_tokens = sum(row["prompt_tokens"] or 0 for row in rows)
        total_prefill_sec = sum(row["prefill_sec"] or 0 for row in rows)
        auto_scored_rows = [row for row in rows if row.get("auto_scored")]
        total_score = sum((row.get("score") or 0) for row in auto_scored_rows)
        total_max_score = sum((row.get("max_score") or 0) for row in auto_scored_rows)

        records.append(
            {
                "model_id": first["model_id"],
                "model_label": first["model_label"],
                "model": first["model"],
                "backend": first["backend"],
                "enable_thinking": bool(first["enable_thinking"]),
                "auto_scored_questions": len(auto_scored_rows),
                "correct_count": int(sum(row["correct"] for row in rows)),
                "full_score_count": int(sum(row["full_score"] for row in rows)),
                "objective_score_sum": total_score,
                "objective_max_score": total_max_score,
                "objective_score_rate": safe_ratio(total_score, total_max_score),
                "avg_latency_sec": round(sum(row["latency_sec"] for row in rows) / len(rows), 2),
                "avg_final_pred_len": round(sum(row["final_pred_len"] for row in rows) / len(rows), 1),
                "avg_prefill_sec": safe_average([row["prefill_sec"] for row in rows]),
                "avg_generation_tps": safe_average([row["generation_tps"] for row in rows]),
                "cold_start_qid": cold_start["qid"],
                "cold_start_prefill_sec": cold_start["prefill_sec"],
                "cold_start_prompt_tps": cold_start["prompt_tps"],
                "steady_state_avg_prefill_sec": safe_average([row["prefill_sec"] for row in steady_state_rows]),
                "steady_state_avg_prompt_tps": safe_average([row["prompt_tps"] for row in steady_state_rows]),
                "steady_state_avg_generation_tps": safe_average([row["generation_tps"] for row in steady_state_rows]),
                "weighted_prompt_tps": safe_ratio(total_prompt_tokens, total_prefill_sec),
            }
        )

    return sorted(
        records,
        key=lambda row: (-(row["objective_score_rate"] or 0), -row["full_score_count"], row["avg_latency_sec"]),
    )


def build_compare_table(model_results: list[list[dict]], tests: list[dict]) -> list[dict]:
    compare = [
        {
            "qid": test["id"],
            "category": test.get("category"),
            "evaluation_type": test.get("evaluation_type", "objective"),
            "gold": get_display_gold(test),
        }
        for test in tests
    ]
    by_qid = {row["qid"]: row for row in compare}

    for rows in model_results:
        model_id = rows[0]["model_id"]
        for row in rows:
            target = by_qid[row["qid"]]
            target[f"final_pred__{model_id}"] = row["final_pred"]
            target[f"correct__{model_id}"] = row["correct"]
            target[f"latency_sec__{model_id}"] = row["latency_sec"]
    return compare


def save_run_artifacts(
    output_dir: Path,
    config_path: Path,
    questions_path: Path | None,
    question_set: dict | None,
    selected_models: list[dict],
    summary: list[dict],
    compare: list[dict],
):
    snapshot = {
        "config_path": str(config_path),
        "questions_path": str(questions_path) if questions_path is not None else None,
        "question_set_name": question_set.get("name") if question_set else "built_in_12",
        "question_set_version": question_set.get("version") if question_set else "built_in",
        "selected_model_ids": [model["id"] for model in selected_models],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(output_dir / "run_config_snapshot.json", snapshot)
    write_csv(output_dir / "summary.csv", summary)
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "compare.csv", compare)
    write_json(output_dir / "compare.json", compare)

