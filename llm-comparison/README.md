# LLM Comparison

这个目录用于做一个小规模的本地 LLM 对比实验。

当前实现全部在 `models_comparison.ipynb` 中，核心流程是：

- 定义 12 道中文推理/计算/代码理解题
- 为每道题构造统一 prompt
- 用 `mlx_lm` 依次运行两个模型
- 从模型输出中提取最后一行 JSON 里的 `final_answer`
- 根据题型规则打分
- 输出逐题对比和汇总结果

## 目录结构

- `models_comparison.ipynb`
  - 主 notebook
- `benchmark_results/results_model_a.json`
  - 模型 A 的逐题结果
- `benchmark_results/results_model_b.json`
  - 模型 B 的逐题结果

## 当前对比的模型

Notebook 里当前写死了两个模型：

- `mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit`
- `mlx-community/Qwen3.5-35B-A3B-4bit`

如果要换模型，直接改 notebook 里的 `MODEL_A` 和 `MODEL_B`。

## 题集

当前题集一共 12 题，覆盖这些类型：

- 基础计算
- 百分比与平均速度
- 排列组合
- 逻辑推理
- Python 代码输出
- 时间复杂度
- 代码 bug 识别
- 简单操作题

每道题包含：

- `id`
- `question`
- `gold`

## 运行依赖

当前 notebook 直接 import 了这些库：

- `json`
- `re`
- `time`
- `pathlib`
- `pandas`
- `mlx_lm`

建议至少准备：

- Python 3.10+
- Jupyter Notebook 或 JupyterLab
- `pandas`
- `mlx_lm`

示例安装：

```bash
pip install pandas mlx-lm jupyter
```

是否能真正跑通，还取决于本机是否能加载对应的 MLX 模型。

## 运行方式

在当前目录启动 Jupyter：

```bash
cd llm-comparison
jupyter notebook
```

然后打开 `models_comparison.ipynb`，按顺序执行。

推荐执行顺序：

1. 定义题集和辅助函数
2. 跑 `MODEL_A`
3. 保存 `results_model_a.json`
4. 跑 `MODEL_B`
5. 保存 `results_model_b.json`
6. 重新加载两个结果文件
7. 计算 `correct`
8. 查看 `summary`
9. 查看逐题 `compare`

## Prompt 约束

Notebook 中的 prompt 要求模型：

- 可以思考，但不要长篇分析
- 有答案后立刻结束
- 最后一行必须输出 JSON
- JSON 格式固定为：

```json
{"final_answer":"你的答案"}
```

`extract_final_answer_json()` 会优先从最后几行里解析这个 JSON；如果失败，再尝试兜底规则。

## 结果文件格式

每条结果会记录：

- `model`
- `qid`
- `question`
- `gold`
- `pred`
- `final_pred`
- `latency_sec`
- `pred_len`
- `final_pred_len`

其中：

- `pred` 是模型完整输出
- `final_pred` 是从输出中抽取出来用于打分的最终答案

## 打分逻辑

打分函数是 `score_row()`。

当前规则不是完全统一的精确匹配，而是按题型分开：

- 精确匹配：
  - `q1 q2 q3 q4 q6 q8 q10 q11`
- 关键词匹配：
  - `q5` 只要包含 `是`
  - `q9` 只要包含与“重复值”相关的关键词
  - `q12` 只要包含 `能`
- 特殊宽松匹配：
  - `q7` 只要完整输出包含 4 段目标结果

所以这里更像是一个手工定义规则的小 benchmark，不是通用评测框架。

## 汇总输出

Notebook 最后会生成两类表：

1. `summary`

- `model`
- `correct_count`
- `avg_latency_sec`
- `avg_final_pred_len`

2. `compare`

- `qid`
- `gold`
- `final_pred_a`
- `correct_a`
- `latency_a`
- `final_pred_b`
- `correct_b`
- `latency_b`

## 已有结果文件

目录中已经有两份样例结果：

- `benchmark_results/results_model_a.json`
- `benchmark_results/results_model_b.json`

当前每份都是 12 条结果，对应 12 道题。

## 已知限制

- 评测数据量很小，结论只能作为快速对比参考
- 题目与打分规则都写死在 notebook 里，不适合直接扩展成大规模 benchmark
- `q5`、`q9`、`q12` 的评分较宽松，可能高估模型表现
- notebook 当前没有做随机种子、重复多次采样或统计显著性分析
- 结果没有导出成图片或正式报告，只在 notebook 中查看 DataFrame

## 后续可扩展方向

- 把题集移到单独的 `tests.json`
- 把模型列表改成配置化
- 把打分规则拆到独立模块
- 增加更多题型和更严格的判分
- 输出 Markdown / HTML 报告
- 支持一次比较多个模型，而不是只比较 A/B 两个
