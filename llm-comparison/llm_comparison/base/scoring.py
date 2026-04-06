import re


def normalize_text(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = normalized.replace("：", ":")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def normalize_compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def normalize_multiline_text(text: str) -> str:
    normalized = text.strip().replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.splitlines()]
    return "\n".join(lines)


def normalize_complexity_text(text: str) -> str:
    normalized = normalize_compact_text(text)
    normalized = normalized.replace("²", "^2").replace("³", "^3")
    normalized = normalized.replace("o(n×n)", "o(n^2)")
    normalized = normalized.replace("o(n*n)", "o(n^2)")
    return normalized


def normalize_date_text(text: str) -> str:
    normalized = normalize_text(str(text))
    match = re.search(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", normalized)
    if not match:
        return normalize_compact_text(str(text))
    year, month, day = (int(part) for part in match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"


def normalize_amount_cny(value) -> str:
    if isinstance(value, (int, float)):
        return str(int(round(float(value))))

    normalized = normalize_compact_text(str(value))
    match = re.match(r"(\d+(?:\.\d+)?)亿元", normalized)
    if match:
        return str(int(round(float(match.group(1)) * 100_000_000)))

    match = re.match(r"(\d+(?:\.\d+)?)万元", normalized)
    if match:
        return str(int(round(float(match.group(1)) * 10_000)))

    match = re.match(r"(\d+(?:\.\d+)?)元", normalized)
    if match:
        return str(int(round(float(match.group(1)))))

    return normalized


def normalize_issue_text(text: str) -> str:
    normalized = normalize_compact_text(str(text))
    normalized = normalized.replace("一开机", "开机")
    normalized = normalized.replace("就报", "报")
    normalized = normalized.replace("错误", "")
    return normalized


def normalize_risk_text(text: str) -> str:
    normalized = normalize_compact_text(str(text))
    concepts = []
    if "监管审批" in normalized or ("审批" in normalized and "监管" in normalized):
        concepts.append("监管审批")
    if "不确定性" in normalized or "不确定" in normalized:
        concepts.append("不确定性")
    return "|".join(concepts) if concepts else normalized


def normalize_json_field_value(key: str, value):
    if value is None:
        return ""
    if key == "date":
        return normalize_date_text(value)
    if key == "amount_cny":
        return normalize_amount_cny(value)
    if key == "issue":
        return normalize_issue_text(value)
    if key == "risk":
        return normalize_risk_text(value)
    return normalize_compact_text(str(value))


def parse_json_candidate(text: str):
    import json

    try:
        return json.loads(text)
    except Exception:
        pass

    candidate = text.strip()
    if candidate.startswith("{'") or candidate.startswith("[{'"):
        try:
            return json.loads(candidate.replace("'", '"'))
        except Exception:
            return None
    return None


def score_row(qid: str, gold: str, pred: str) -> int:
    gold_normalized = normalize_text(gold)
    pred_normalized = normalize_text(pred)

    exact_match_ids = {"q1", "q2", "q3", "q4", "q6", "q8", "q10", "q11"}
    if qid in exact_match_ids:
        return int(gold_normalized == pred_normalized)
    if qid == "q5":
        return int("是" in pred_normalized)
    if qid == "q9":
        keys = ["重复", "第二大不同", "没有处理重复值"]
        return int(any(key in pred_normalized for key in keys))
    if qid == "q12":
        return int("能" in pred_normalized)
    if qid == "q7":
        want_parts = ["[1]", "[1, 2]", "[3]", "[1, 2, 4]"]
        return int(all(part in pred for part in want_parts))
    return 0


def score_exact_match(question: dict, pred: str):
    return int(normalize_text(question.get("gold", "")) == normalize_text(pred)), 1


def score_multiline_exact_match(question: dict, pred: str):
    return int(normalize_multiline_text(question.get("gold", "")) == normalize_multiline_text(pred)), 1


def score_complexity_alias_match(question: dict, pred: str):
    aliases = [question.get("gold", "")]
    aliases.extend(question.get("accepted_aliases", []))
    normalized_pred = normalize_complexity_text(pred)
    normalized_aliases = {normalize_complexity_text(alias) for alias in aliases if alias}
    return int(normalized_pred in normalized_aliases), 1


def score_contains_any_keyword(question: dict, pred: str):
    normalized_pred = normalize_compact_text(pred)
    keywords = [normalize_compact_text(keyword) for keyword in question.get("gold_keywords", [])]
    if any(keyword in normalized_pred for keyword in keywords):
        return 1, 1

    if question.get("id") == "bq06":
        mentions_accumulate = "累加" in pred
        mentions_reset = any(token in pred for token in ["重置", "赋值", "置为1", "计数都为1", "计数相同"])
        return int(mentions_accumulate and mentions_reset), 1
    return 0, 1


def score_json_field_exact(question: dict, pred: str):
    target = question.get("gold_json", {})
    data = parse_json_candidate(pred)
    if not isinstance(data, dict):
        return 0, 3

    score = 1
    required_fields = set(target.keys())
    if not required_fields.issubset(data.keys()):
        return score, 3
    score += 1

    field_values_ok = all(
        normalize_json_field_value(key, data.get(key, "")) == normalize_json_field_value(key, value)
        for key, value in target.items()
    )
    if field_values_ok:
        score += 1
    return score, 3


def score_constraint_check(question: dict, pred: str):
    constraints = question.get("gold_constraints", {})
    score = 0

    exact_length_chars = constraints.get("exact_length_chars")
    if exact_length_chars is None or len(pred) == exact_length_chars:
        score += 1

    forbid_punctuation = constraints.get("forbid_punctuation")
    punctuation_pattern = r"[，。！？；：、“”‘’,.!?;:()（）\-]"
    if not forbid_punctuation or not re.search(punctuation_pattern, pred):
        score += 1

    must_cover = constraints.get("must_cover", [])
    if all(token in pred for token in must_cover):
        score += 1
    return score, 3


def score_grounded_binary(question: dict, pred: str):
    score = 0
    normalized_pred = normalize_compact_text(pred)
    keywords = [normalize_compact_text(keyword) for keyword in question.get("gold_keywords", [])]
    if all(keyword in normalized_pred for keyword in keywords):
        score += 1
    if len(pred.strip()) > 0:
        score += 1
    forbidden = question.get("forbidden_hallucinations", [])
    if not any(token in pred for token in forbidden):
        score += 1
    return score, 3


def score_with_rule(question: dict, pred: str):
    evaluation_type = question.get("evaluation_type", "objective")
    if evaluation_type != "objective":
        return None, None, False

    rule = question.get("scoring_rule")
    if not rule:
        return score_row(question["id"], question.get("gold", ""), pred), 1, True

    handlers = {
        "exact_match": score_exact_match,
        "multiline_exact_match": score_multiline_exact_match,
        "complexity_alias_match": score_complexity_alias_match,
        "contains_any_keyword": score_contains_any_keyword,
        "json_field_exact": score_json_field_exact,
        "constraint_check": score_constraint_check,
        "grounded_binary": score_grounded_binary,
    }
    if rule not in handlers:
        return None, None, False

    score, max_score = handlers[rule](question, pred)
    return score, max_score, True


def add_scores(results: list[dict], questions_by_id: dict[str, dict]) -> list[dict]:
    scored = []
    for row in results:
        item = dict(row)
        question = questions_by_id[item["qid"]]
        score, max_score, auto_scored = score_with_rule(question, item["final_pred"])
        item["score"] = score
        item["max_score"] = max_score
        item["auto_scored"] = auto_scored
        item["correct"] = int(auto_scored and max_score == 1 and score == 1)
        item["full_score"] = int(auto_scored and max_score is not None and score == max_score)
        scored.append(item)
    return scored

