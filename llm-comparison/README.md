# LLM Comparison

本项目用于本地 MLX 模型对比评测，当前覆盖 5 个模型与 3 套测试：

- `classic12`
  - 最早的 12 道客观题，偏数学、逻辑和代码理解
- `base`
  - 扩展基础能力题，包含客观题与 `rubric` 人工评分题
- `tool`
  - 工具使用题，评估工具选择、参数构造、多步调用与失败恢复

项目默认采用命令行方式运行，notebook 仅作为参考与调试入口。

## 项目结构

- `llm_comparison/`
  - 项目主包
  - `common/`：通用控制台、文件、模型加载、运行时与统计逻辑
  - `base/`：基础题 prompt、评分、汇总与 runner
  - `tooling/`：工具题 prompt、工具运行时、评分、汇总与 runner
  - `templates/`：`rubric` 人工评分模板生成
- `compare_models.py`
  - 基础题入口脚本
- `run_tool_benchmark.py`
  - 工具题入口脚本
- `generate_rubric_template.py`
  - `rubric` 模板生成入口脚本
- `config/models_config.json`
  - 模型配置
- `benchmarks/questions/`
  - 题库定义
- `benchmarks/scoring/`
  - 评分规则
- `fixtures/tooling/mock_project/`
  - 工具题 mock 数据环境
- `benchmark_results/five_models_20260406/`
  - 已整理的 5 模型结果包
- `notebook/`
  - 参考 notebook 与早期 notebook 样例结果
- `obsolete/`
  - 已被整理包替代的旧结果与旧评分文件

## 模型配置

默认模型配置位于 [models_config.json](/Users/jenslewie/github/llm-learning/llm-comparison/config/models_config.json)，当前包含：

- `qwen35_27b_opus_distilled`
- `qwen35_35b_a3b`
- `qwen35_27b_6bit`
- `gemma4_26b_a4b`
- `gemma4_31b`

每个模型均显式配置：

- `backend`
- `max_tokens`
- `enable_thinking`

当前默认均为 `enable_thinking = false`。

## 运行环境

建议环境：

- Python 3.10+
- `mlx_lm`
- `mlx_vlm`

示例安装：

```bash
pip install mlx-lm mlx-vlm
```

## 使用方式

运行最早的 12 题：

```bash
cd llm-comparison
python compare_models.py
```

运行基础题库：

```bash
python compare_models.py --questions benchmarks/questions/base_questions.json
```

运行工具题库：

```bash
python run_tool_benchmark.py
```

仅运行部分模型：

```bash
python compare_models.py --models qwen35_27b_6bit gemma4_31b
python run_tool_benchmark.py --models qwen35_27b_6bit gemma4_31b
```

复用已有结果目录继续补跑：

```bash
python compare_models.py --run-name my_run --skip-existing
python run_tool_benchmark.py --run-name my_tool_run --skip-existing
```

生成 `rubric` 人工评分模板：

```bash
python generate_rubric_template.py \
  --questions benchmarks/questions/base_questions.json \
  --scoring benchmarks/scoring/base_scoring.json \
  --results benchmark_results/<run_name>/results_<model_id>.json \
  --output manual_scoring.csv
```

## 结果输出

常规运行结果输出到：

```bash
benchmark_results/<run_name>/
```

典型文件包括：

- `results_<model_id>.json`
- `summary.csv`
- `compare.csv`
- `run_config_snapshot.json`

其中：

- `summary.csv`
  - 模型级汇总指标
- `compare.csv`
  - 逐题横向对比
- `run_config_snapshot.json`
  - 本次运行所用配置与题库快照

## 当前整理结果

已整理好的 5 模型结果位于：

- [five_models_20260406](/Users/jenslewie/github/llm-learning/llm-comparison/benchmark_results/five_models_20260406)

目录结构如下：

- [1-classic](/Users/jenslewie/github/llm-learning/llm-comparison/benchmark_results/five_models_20260406/1-classic)
- [2-base](/Users/jenslewie/github/llm-learning/llm-comparison/benchmark_results/five_models_20260406/2-base)
- [3-tool](/Users/jenslewie/github/llm-learning/llm-comparison/benchmark_results/five_models_20260406/3-tool)
- [overview](/Users/jenslewie/github/llm-learning/llm-comparison/benchmark_results/five_models_20260406/overview)

跨 3 套测试的总表见：

- [combined_suite_scores.csv](/Users/jenslewie/github/llm-learning/llm-comparison/benchmark_results/five_models_20260406/overview/combined_suite_scores.csv)
- [combined_suite_scores.md](/Users/jenslewie/github/llm-learning/llm-comparison/benchmark_results/five_models_20260406/overview/combined_suite_scores.md)

当前汇总如下：

| 模型 | backend | 12题 | 12题耗时 | 基础题 | 基础题耗时 | 工具题 | 工具题耗时 | 综合均值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma-4-31B (4bit) | `mlx_vlm` | `12/12` | `4.47s` | `42/44` | `4.86s` | `39/42` | `10.64s` | `96.10%` |
| Qwen3.5-27B Claude-4.6 Opus Distilled (4bit) | `mlx_lm` | `12/12` | `19.38s` | `37/44` | `23.73s` | `38/42` | `23.89s` | `91.52%` |
| Gemma-4-26B-A4B (4bit) | `mlx_vlm` | `10/12` | `1.50s` | `42/44` | `1.97s` | `32/42` | `2.50s` | `84.99%` |
| Qwen3.5-27B (6bit) | `mlx_lm` | `9/12` | `2.79s` | `39/44` | `4.06s` | `37/42` | `8.36s` | `83.91%` |
| Qwen3.5-35B-A3B (4bit) | `mlx_lm` | `9/12` | `0.73s` | `31/44` | `1.02s` | `39/42` | `4.12s` | `79.44%` |

说明：

- `综合均值` 为 3 套测试命中率的简单平均
- qwen 三个模型运行于 `mlx_lm`，gemma 两个模型运行于 `mlx_vlm`
- 因 runtime 不完全一致，性能指标更适合作为实际部署口径下的参考
