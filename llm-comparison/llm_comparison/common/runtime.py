import gc
import importlib

from llm_comparison.common.console import log


def release_model_resources(model=None, processor=None):
    del model
    del processor
    gc.collect()

    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def get_backend_bundle(backend: str):
    if backend == "mlx_lm":
        log(f"Importing backend module: {backend}")
        module = importlib.import_module("mlx_lm")
        return {
            "load": module.load,
            "generate": module.generate,
            "stream_generate": module.stream_generate,
        }

    if backend == "mlx_vlm":
        log(f"Importing backend module: {backend}")
        module = importlib.import_module("mlx_vlm")
        prompt_utils = importlib.import_module("mlx_vlm.prompt_utils")
        utils = importlib.import_module("mlx_vlm.utils")
        return {
            "load": module.load,
            "generate": module.generate,
            "stream_generate": module.stream_generate,
            "apply_chat_template": prompt_utils.apply_chat_template,
            "load_config": utils.load_config,
        }

    raise ValueError(f"Unsupported backend: {backend}")


def build_generation_prompt(backend: str, backend_bundle, processor, config, messages, enable_thinking: bool):
    if backend == "mlx_lm":
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=False,
            enable_thinking=enable_thinking,
        )

    if backend == "mlx_vlm":
        return backend_bundle["apply_chat_template"](
            processor,
            config,
            messages,
            add_generation_prompt=True,
            num_images=0,
            enable_thinking=enable_thinking,
        )

    raise ValueError(f"Unsupported backend: {backend}")


def extract_stream_metrics(last_response):
    prompt_tokens = getattr(last_response, "prompt_tokens", None)
    generation_tokens = getattr(last_response, "generation_tokens", None)
    prompt_tps = getattr(last_response, "prompt_tps", None)
    generation_tps = getattr(last_response, "generation_tps", None)
    peak_memory = getattr(last_response, "peak_memory", None)

    prefill_sec = None
    if prompt_tokens and prompt_tps:
        prefill_sec = round(prompt_tokens / prompt_tps, 4)

    decode_sec = None
    if generation_tokens and generation_tps:
        decode_sec = round(generation_tokens / generation_tps, 4)

    return {
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prompt_tps": round(prompt_tps, 4) if prompt_tps is not None else None,
        "generation_tps": round(generation_tps, 4) if generation_tps is not None else None,
        "prefill_sec": prefill_sec,
        "decode_sec": decode_sec,
        "peak_memory_gb": round(peak_memory, 4) if peak_memory is not None else None,
    }

