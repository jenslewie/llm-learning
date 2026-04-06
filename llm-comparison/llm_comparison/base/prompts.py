import json
import re
from pathlib import Path

from llm_comparison.base.defaults import DEFAULT_TESTS
from llm_comparison.common.files import load_json


def build_prompt(question: str) -> str:
    return f"""你将回答一道推理题。

要求：
1. 不要输出思考过程。
2. 如果得出答案，请立刻结束。
3. 最后一行必须输出一个 JSON 对象。
4. JSON 格式必须是：{{"final_answer":"你的答案"}}
5. 不要在最后一行输出其他内容。

题目如下：
{question}
"""


def extract_final_answer_json(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "final_answer" in obj:
                final_answer = obj["final_answer"]
                if isinstance(final_answer, (dict, list)):
                    return json.dumps(final_answer, ensure_ascii=False)
                return str(final_answer).strip()
        except Exception:
            pass

    matches = re.findall(r"(?im)^\s*FINAL\s*:\s*(.+?)\s*$", text)
    if matches:
        return matches[-1].strip()

    return lines[-1] if lines else ""


def get_question_text(question: dict) -> str:
    return question.get("question") or question.get("prompt") or ""


def get_display_gold(question: dict):
    if "gold" in question:
        return question["gold"]
    if "gold_json" in question:
        return json.dumps(question["gold_json"], ensure_ascii=False, sort_keys=True)
    if "gold_keywords" in question:
        return " / ".join(question["gold_keywords"])
    return ""


def load_questions(question_path: Path | None) -> tuple[dict | None, list[dict]]:
    if question_path is None:
        return None, DEFAULT_TESTS

    payload = load_json(question_path)
    questions = payload.get("questions", [])
    if not questions:
        raise ValueError(f"No questions found in file: {question_path}")

    ids = [question["id"] for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Question ids in question file must be unique.")
    return payload, questions
