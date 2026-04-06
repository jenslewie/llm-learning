from pathlib import Path

from llm_comparison.common.files import load_json


def load_models_config(config_path: Path) -> list[dict]:
    config = load_json(config_path)
    models = config.get("models", [])
    if not models:
        raise ValueError(f"No models found in config: {config_path}")

    ids = [model["id"] for model in models]
    if len(ids) != len(set(ids)):
        raise ValueError("Model ids in config must be unique.")
    return models


def select_models(models: list[dict], selected_ids: list[str] | None) -> list[dict]:
    if not selected_ids:
        return models

    wanted = set(selected_ids)
    selected = [model for model in models if model["id"] in wanted]
    present_ids = {model["id"] for model in selected}
    missing = [model_id for model_id in selected_ids if model_id not in present_ids]
    if missing:
        raise ValueError(f"Unknown model ids: {', '.join(missing)}")
    return selected

