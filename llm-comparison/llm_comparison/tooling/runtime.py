import json
import math
import time
from pathlib import Path


FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "tooling" / "mock_project"

ORDERS_DATA = [
    {"order_id": "ORD-9001", "status": "processing", "amount": 9200},
    {"order_id": "ORD-9002", "status": "delayed", "amount": 15800},
    {"order_id": "ORD-9003", "status": "delayed", "amount": 24400},
    {"order_id": "ORD-9004", "status": "completed", "amount": 30100},
]
TICKET_DATA = {
    "T-1042": {
        "ticket_id": "T-1042",
        "status": "blocked",
        "owner": "payments",
        "summary": "等待上游支付渠道修复",
        "eta": "2026-04-07",
    }
}
DAILY_NEW_USERS = {"2026-04-05": 128}
SALES_REPORT = {"2026-04-01": 128700}

TOOL_SPECS = {
    "calculator": {"description": '计算数学表达式。参数: {"expression": "(287 * 913 - 17451) / 7"}'},
    "file_search": {"description": '在 mock_project 目录中按内容关键词或文件名模式搜索。参数可选: {"query": "PAYMENT_TIMEOUT_MS"} 或 {"pattern": "*.log"}'},
    "file_read": {"description": '读取 mock_project 下的文件内容。参数: {"path": "config/app.env"}'},
    "json_query": {"description": '查询内置 orders 数据集。参数示例: {"dataset": "orders", "filters": {"status": "delayed"}, "sort_by": "amount", "sort_order": "desc", "limit": 1}'},
    "http_request": {"description": '查询内置工单 API。参数示例: {"ticket_id": "T-1042"} 或 {"path": "/tickets/T-1042"}'},
    "db_query": {"description": '查询内置数据库统计。参数示例: {"db": "analytics", "metric": "new_users", "date": "2026-04-05"}'},
    "sales_report_api": {"description": '查询销售额报表。参数必须包含 date，例如 {"date": "2026-04-01"}'},
    "python_runner": {"description": '执行受限的数值任务。当前仅支持 primality_check。参数示例: {"task": "primality_check", "n": 9973}'},
    "log_counter": {"description": '统计日志文件中某个词的出现次数。参数示例: {"paths": ["logs/api.log", "logs/web.log"], "term": "error"}'},
}


def list_tool_specs(tool_names: list[str]) -> list[dict]:
    return [{"name": tool_name, **TOOL_SPECS[tool_name]} for tool_name in tool_names]


def _safe_eval_expression(expression: str):
    allowed_chars = set("0123456789+-*/(). %")
    if any(char not in allowed_chars for char in expression):
        raise ValueError("Unsupported characters in expression")
    return eval(expression, {"__builtins__": {}}, {})


def _search_files(query: str | None = None, pattern: str | None = None) -> list[str]:
    results = []
    for path in FIXTURES_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = str(path.relative_to(FIXTURES_ROOT))
        if pattern and path.match(pattern):
            results.append(relative)
            continue
        if query:
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue
            if query in content or query in relative:
                results.append(relative)
    return sorted(results)


def _read_file(path: str) -> str:
    target = (FIXTURES_ROOT / path).resolve()
    if FIXTURES_ROOT.resolve() not in target.parents and target != FIXTURES_ROOT.resolve():
        raise ValueError("Path escapes fixture root")
    return target.read_text(encoding="utf-8")


def _json_query(arguments: dict):
    if arguments.get("dataset") != "orders":
        raise ValueError("Unsupported dataset")
    rows = list(ORDERS_DATA)
    for key, value in arguments.get("filters", {}).items():
        rows = [row for row in rows if str(row.get(key)) == str(value)]
    sort_by = arguments.get("sort_by")
    if sort_by:
        reverse = str(arguments.get("sort_order", "asc")).lower() == "desc"
        rows = sorted(rows, key=lambda row: row.get(sort_by, 0), reverse=reverse)
    return rows[: int(arguments.get("limit", len(rows)))]


def _http_request(arguments: dict):
    ticket_id = arguments.get("ticket_id")
    if not ticket_id and arguments.get("path"):
        ticket_id = str(arguments["path"]).rstrip("/").split("/")[-1]
    if ticket_id not in TICKET_DATA:
        raise ValueError("Ticket not found")
    return TICKET_DATA[ticket_id]


def _db_query(arguments: dict):
    if arguments.get("db") != "analytics":
        raise ValueError("Unknown db")
    if arguments.get("metric") != "new_users":
        raise ValueError("Unsupported metric")
    date = arguments.get("date")
    if date not in DAILY_NEW_USERS:
        raise ValueError("No data for date")
    return {"db": "analytics", "metric": "new_users", "date": date, "value": DAILY_NEW_USERS[date]}


def _sales_report_api(arguments: dict):
    if "date" not in arguments:
        raise ValueError("Missing required parameter: date")
    date = arguments["date"]
    if date not in SALES_REPORT:
        raise ValueError("No sales report for date")
    return {"date": date, "revenue": SALES_REPORT[date]}


def _python_runner(arguments: dict):
    if arguments.get("task") != "primality_check":
        raise ValueError("Unsupported python task")
    n = int(arguments["n"])
    if n < 2:
        return {"n": n, "is_prime": False}
    for divisor in range(2, int(math.isqrt(n)) + 1):
        if n % divisor == 0:
            return {"n": n, "is_prime": False, "factor": divisor}
    return {"n": n, "is_prime": True}


def _log_counter(arguments: dict):
    term = str(arguments.get("term", "")).lower()
    counts = {}
    for relative_path in arguments.get("paths", []):
        content = _read_file(relative_path)
        counts[relative_path] = sum(1 for line in content.splitlines() if term in line.lower())
    return counts


def execute_tool(tool_name: str, arguments: dict):
    if tool_name not in TOOL_SPECS:
        raise ValueError(f"Unsupported tool: {tool_name}")
    handlers = {
        "calculator": lambda args: {"result": int(result) if isinstance((result := _safe_eval_expression(str(args['expression']))), float) and result.is_integer() else result},
        "file_search": lambda args: {"matches": _search_files(args.get("query"), args.get("pattern"))},
        "file_read": lambda args: {"path": args["path"], "content": _read_file(args["path"])},
        "json_query": lambda args: {"rows": _json_query(args)},
        "http_request": _http_request,
        "db_query": _db_query,
        "sales_report_api": _sales_report_api,
        "python_runner": _python_runner,
        "log_counter": lambda args: {"counts": _log_counter(args)},
    }
    return handlers[tool_name](arguments)


def run_tool_and_trace(tool_name: str, arguments: dict):
    started_at = time.time()
    try:
        output = execute_tool(tool_name, arguments)
        success = True
        error_message = ""
    except Exception as exc:
        output = None
        success = False
        error_message = str(exc)
    finished_at = time.time()
    trace = {
        "tool_name": tool_name,
        "arguments": arguments,
        "call_started_at": started_at,
        "call_finished_at": finished_at,
        "latency_sec": round(finished_at - started_at, 4),
        "success": success,
        "error_message": error_message,
        "output_excerpt": json.dumps(output, ensure_ascii=False)[:500] if output is not None else "",
    }
    return output, trace
