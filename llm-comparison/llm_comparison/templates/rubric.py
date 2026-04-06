import argparse
from pathlib import Path

from llm_comparison.common.files import load_json, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manual scoring template for rubric questions.")
    parser.add_argument("--questions", required=True, help="Path to the question-set JSON file.")
    parser.add_argument("--scoring", required=True, help="Path to the scoring JSON file.")
    parser.add_argument("--results", nargs="*", help="Optional result JSON files.")
    parser.add_argument("--output", required=True, help="Path to the output CSV template.")
    return parser.parse_args()


def get_rubric_questions(question_set: dict) -> list[dict]:
    return [question for question in question_set.get("questions", []) if question.get("evaluation_type") == "rubric"]


def build_base_row(question: dict, rubric: dict) -> dict:
    anchors = rubric.get("anchors", {})
    anchor_lines = [f"{score}: {text}" for score, text in sorted(anchors.items(), key=lambda item: int(item[0]))]
    dimensions = rubric.get("dimensions", [])
    row = {
        "run_name": "",
        "result_file": "",
        "model_id": "",
        "model_label": "",
        "qid": question["id"],
        "category": question.get("category", ""),
        "rubric_id": question.get("rubric_id", ""),
        "score_range": f"{rubric.get('score_range', ['?', '?'])[0]}-{rubric.get('score_range', ['?', '?'])[1]}",
        "dimension_count": len(dimensions),
        "dimensions": " | ".join(dimensions),
        "anchor_points": " | ".join(question.get("anchor_points", [])),
        "rubric_anchors": " || ".join(anchor_lines),
        "prompt": question.get("prompt", ""),
        "output_contract": question.get("output_contract", ""),
        "final_pred": "",
        "human_score_overall": "",
        "judge_note": "",
    }
    for idx, dimension in enumerate(dimensions, 1):
        row[f"dimension_{idx}_name"] = dimension
        row[f"dimension_{idx}_score"] = ""
        row[f"dimension_{idx}_note"] = ""
    return row


def build_rows(question_set: dict, scoring: dict, result_paths: list[Path]) -> list[dict]:
    rubrics = scoring.get("rubrics", {})
    questions = get_rubric_questions(question_set)
    if not result_paths:
        return [build_base_row(question, rubrics[question["rubric_id"]]) for question in questions]

    rows = []
    questions_by_id = {question["id"]: question for question in questions}
    for result_path in result_paths:
        results = load_json(result_path)
        run_name = result_path.parent.name
        for item in results:
            qid = item.get("qid")
            if qid not in questions_by_id:
                continue
            question = questions_by_id[qid]
            row = build_base_row(question, rubrics[question["rubric_id"]])
            row["run_name"] = run_name
            row["result_file"] = str(result_path)
            row["model_id"] = item.get("model_id", "")
            row["model_label"] = item.get("model_label", "")
            row["final_pred"] = item.get("final_pred", "")
            rows.append(row)
    return rows


def main():
    args = parse_args()
    question_set = load_json(Path(args.questions).resolve())
    scoring = load_json(Path(args.scoring).resolve())
    result_paths = [Path(path).resolve() for path in (args.results or [])]
    output_path = Path(args.output).resolve()

    rows = build_rows(question_set, scoring, result_paths)
    write_csv(output_path, rows)
    print(f"Generated rubric template: {output_path}")
    print(f"Rows: {len(rows)}")

