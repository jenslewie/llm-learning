# Five-Model Benchmark Summary

- Generated at: 2026-04-07T09:17:03
- Source dirs: `1-classic`, `2-base`, `3-tool`

| 模型 | backend | 12题 | 12题耗时 | 基础题 | 基础题耗时 | 工具题 | 工具题耗时 | 综合均值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-4-31B (4bit) | mlx_vlm | 12/12 | 4.90s | 42/44 | 5.20s | 39/42 | 10.21s | 96.10% |
| Qwen3.5-27B Claude-4.6 Opus Distilled (4bit) | mlx_lm | 12/12 | 18.83s | 37/44 | 21.53s | 38/42 | 21.70s | 91.52% |
| Gemma-4-26B-A4B (4bit) | mlx_vlm | 10/12 | 2.55s | 42/44 | 3.69s | 34/42 | 3.30s | 86.58% |
| Qwen3.5-27B (6bit) | mlx_lm | 9/12 | 5.12s | 38/44 | 5.15s | 37/42 | 11.28s | 83.15% |
| Qwen3.5-35B-A3B (4bit) | mlx_lm | 9/12 | 2.52s | 32/44 | 4.80s | 39/42 | 3.21s | 80.20% |
