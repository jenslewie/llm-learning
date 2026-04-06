from llm_comparison.base.scoring import parse_json_candidate
from llm_comparison.tooling.prompts import normalize_tool_answer


def sequence_prefix_match(actual: list[str], expected: list[str]) -> bool:
    if not expected:
        return True
    if len(actual) < len(expected):
        return False
    return actual[: len(expected)] == expected


def score_tool_question(question: dict, row: dict) -> dict:
    traces = row.get("tool_trace", [])
    trace_tools = [trace["tool_name"] for trace in traces]
    expected_tools = question.get("expected_tool_sequence", [])
    allowed_tools = set(question.get("allowed_tools", []))
    final_pred = str(row.get("final_pred", ""))
    scoring_rule = question.get("scoring_rule")

    def keyword_match() -> bool:
        keywords = [normalize_tool_answer(keyword) for keyword in question.get("gold_keywords", [])]
        normalized_pred = normalize_tool_answer(final_pred)
        return all(keyword in normalized_pred for keyword in keywords)

    def final_exact_match() -> bool:
        return normalize_tool_answer(final_pred) == normalize_tool_answer(question.get("gold", ""))

    def final_json_match() -> bool:
        target = question.get("gold_json", {})
        data = parse_json_candidate(final_pred)
        if not isinstance(data, dict):
            return False
        return all(normalize_tool_answer(data.get(key, "")) == normalize_tool_answer(value) for key, value in target.items())

    if scoring_rule == "single_tool_objective":
        subscores = {
            "used_expected_tool": int(any(tool in expected_tools for tool in trace_tools)),
            "parameter_valid": int(any(trace["success"] for trace in traces)),
            "tool_call_success": int(any(trace["success"] for trace in traces)),
            "final_answer_correct": 2 if final_exact_match() else 0,
        }
        max_score = 5
    elif scoring_rule == "tool_trace_plus_answer":
        subscores = {
            "tool_sequence_reasonable": int(sequence_prefix_match(trace_tools, expected_tools)),
            "parameter_valid": int(all(trace["success"] for trace in traces) and all(trace["tool_name"] in allowed_tools for trace in traces)),
            "tool_result_used_correctly": int(any(trace["success"] for trace in traces) and keyword_match()),
            "final_answer_correct": 2 if keyword_match() else 0,
        }
        max_score = 5
    elif scoring_rule == "tool_trace_plus_json":
        data = parse_json_candidate(final_pred)
        subscores = {
            "tool_sequence_reasonable": int(sequence_prefix_match(trace_tools, expected_tools)),
            "parameter_valid": int(all(trace["success"] for trace in traces) and all(trace["tool_name"] in allowed_tools for trace in traces)),
            "json_valid": int(isinstance(data, dict)),
            "json_fields_complete": int(isinstance(data, dict) and set(question.get("gold_json", {}).keys()).issubset(data.keys())),
            "content_correct": 2 if final_json_match() else 0,
        }
        max_score = 6
    elif scoring_rule == "recovery_required":
        recovered = len(traces) >= 2 and (not traces[0]["success"]) and any(trace["success"] for trace in traces[1:])
        subscores = {
            "attempted_tool_call": int(len(traces) >= 1),
            "recognized_failure": int(any(not trace["success"] for trace in traces)),
            "recovered_with_correct_params": 2 if recovered else 0,
            "final_answer_correct": 2 if final_exact_match() else 0,
        }
        max_score = 6
    elif scoring_rule == "tool_selection_judgment":
        tool_required = bool(question.get("tool_required", True))
        used_tool = len(traces) > 0
        judged_correctly = (tool_required and used_tool) or ((not tool_required) and (not used_tool))
        subscores = {
            "judged_need_for_tool_correctly": 2 if judged_correctly else 0,
            "if_used_tool_then_reasonable": int((not used_tool) or all(trace["tool_name"] in allowed_tools for trace in traces)),
            "final_answer_correct": 2 if keyword_match() else 0,
        }
        max_score = 5
    elif scoring_rule == "tool_trace_plus_report":
        counts_ok = False
        for trace in traces:
            if trace["tool_name"] == "log_counter" and trace["success"]:
                output = parse_json_candidate(trace["output_excerpt"])
                if isinstance(output, dict) and output.get("counts", {}) == {"logs/api.log": 2, "logs/worker.log": 1, "logs/web.log": 3}:
                    counts_ok = True
        subscores = {
            "used_required_tools": int(sequence_prefix_match(trace_tools, expected_tools)),
            "counts_correct": 2 if counts_ok else 0,
            "summary_uses_tool_results": int(keyword_match()),
            "report_clear": int(len(final_pred.strip()) > 0),
        }
        max_score = 5
    else:
        raise ValueError(f"Unsupported tool scoring rule: {scoring_rule}")

    score = sum(subscores.values())
    return {
        "score": score,
        "max_score": max_score,
        "tool_trace_tools": trace_tools,
        "subscores": subscores,
        "full_score": int(score == max_score),
        "end_to_end_success": int(score > 0 and (subscores.get("final_answer_correct", 0) > 0 or subscores.get("content_correct", 0) > 0)),
    }

