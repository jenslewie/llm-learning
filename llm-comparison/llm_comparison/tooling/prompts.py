import json
import re
from pathlib import Path

from llm_comparison.base.scoring import normalize_compact_text, parse_json_candidate
from llm_comparison.common.files import load_json
from llm_comparison.tooling.runtime import FIXTURES_ROOT, list_tool_specs


def load_tool_questions(question_path: Path) -> tuple[dict, list[dict]]:
    payload = load_json(question_path)
    questions = payload.get("questions", [])
    if not questions:
        raise ValueError(f"No questions found in file: {question_path}")

    ids = [question["id"] for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Question ids in question file must be unique.")
    return payload, questions


def build_tool_prompt(question: dict) -> str:
    tool_specs = list_tool_specs(question.get("allowed_tools", []))
    return (
        "你正在完成一道工具使用题。请根据任务决定是否调用工具。\n\n"
        "规则：\n"
        "1. 不要输出思考过程。\n"
        "2. 每一步只能输出一个 JSON 对象，不要输出 markdown。\n"
        '3. 如果需要调用工具，输出：{"action":"tool","tool_name":"工具名","arguments":{...}}\n'
        '4. 如果已经可以回答，输出：{"action":"final","final_answer":"你的最终答案"}\n'
        "5. 只能使用题目允许的工具。\n"
        "6. 工具结果会在下一轮作为新消息提供给你。\n\n"
        f"可用工具：\n{json.dumps(tool_specs, ensure_ascii=False, indent=2)}\n\n"
        f"mock 工具工作目录：{FIXTURES_ROOT}\n\n"
        f"任务：\n{question['task']}\n"
    )


def parse_action_json(text: str) -> dict:
    candidate = parse_json_candidate(text)
    if isinstance(candidate, dict) and "action" in candidate:
        return candidate

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        candidate = parse_json_candidate(line)
        if isinstance(candidate, dict) and "action" in candidate:
            return candidate

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidate = parse_json_candidate(match.group(0))
        if isinstance(candidate, dict) and "action" in candidate:
            return candidate

    return {"action": "final", "final_answer": lines[-1] if lines else text.strip()}


def maybe_inject_first_call_argument_error(question: dict, tool_name: str, arguments: dict, traces: list[dict]) -> tuple[dict, dict | None]:
    config = question.get("force_first_tool_argument_error")
    if not config or traces or tool_name != config.get("tool_name"):
        return arguments, None

    source_key = config.get("source_key", "date")
    wrong_key = config.get("wrong_key", "day")
    actual_value = arguments.get(source_key, config.get("fallback_value"))

    injected_arguments = dict(arguments)
    injected_arguments.pop(source_key, None)
    injected_arguments[wrong_key] = actual_value
    return injected_arguments, {
        "injected": True,
        "reason": "force_first_tool_argument_error",
        "model_arguments": arguments,
        "executed_arguments": injected_arguments,
    }


def normalize_tool_answer(text: str) -> str:
    return normalize_compact_text(str(text))

