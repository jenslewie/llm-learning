# Five-Model Benchmark Summary

- Generated at: 2026-04-06T21:32:02
- Source dirs: `1-classic`, `2-base`, `3-tool`

| 模型 | backend | 12题 | 12题耗时 | 基础题 | 基础题耗时 | 工具题 | 工具题耗时 | 综合均值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-4-31B (4bit) | mlx_vlm | 12/12 | 4.47s | 42/44 | 4.86s | 39/42 | 10.64s | 96.10% |
| Qwen3.5-27B Claude-4.6 Opus Distilled (4bit) | mlx_lm | 12/12 | 19.38s | 37/44 | 23.73s | 38/42 | 23.89s | 91.52% |
| Gemma-4-26B-A4B (4bit) | mlx_vlm | 10/12 | 1.50s | 42/44 | 1.97s | 32/42 | 2.50s | 84.99% |
| Qwen3.5-27B (6bit) | mlx_lm | 9/12 | 2.79s | 39/44 | 4.06s | 37/42 | 8.36s | 83.91% |
| Qwen3.5-35B-A3B (4bit) | mlx_lm | 9/12 | 0.73s | 31/44 | 1.02s | 39/42 | 4.12s | 79.44% |
