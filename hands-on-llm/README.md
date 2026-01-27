# Hands-on Large Language Models - Learning Notes

This directory contains learning notes and code examples from "Hands-on Large Language Models" by Jay Alammar and Maarten Grootendorst.

## 📖 About This Book

This book is divided into three parts with 12 chapters, covering a complete LLM learning path from basic concepts to advanced training techniques.

## 🖥️ Environment

- **OS**: macOS (Apple Silicon - M4 Max)
- **Python**: 3.11+
- **Main Dependencies**: transformers, torch, accelerate

### Quick Start

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate llm-learning-phi-3-mini

# Or create environment manually
conda create -n llm-learning-phi-3-mini python=3.11 -y
conda activate llm-learning-phi-3-mini
pip install torch transformers==4.41.0 accelerate
```

### ⚠️ Apple Silicon Users

Please refer to the [Apple Silicon Setup Guide](../docs/setup/apple-silicon-setup.md) for platform-specific configuration requirements.

## 📚 Chapter Index

### Part I: Understanding Language Models

| Chapter | Topic | Status |
|---------|-------|--------|
| [Chapter 1](./chapter-01-introduction/) | Introduction to LLMs | ✅ Complete |
| [Chapter 2](./chapter-02-tokens-embeddings/) | Tokens and Embeddings | 🚧 In Progress |
| [Chapter 3](./chapter-03-transformer-architecture/) | Inside Transformer Architecture | ⏳ Pending |

### Part II: Using Pretrained Models

| Chapter | Topic | Status |
|---------|-------|--------|
| [Chapter 4](./chapter-04-text-classification/) | Text Classification | ⏳ Pending |
| [Chapter 5](./chapter-05-clustering-topic-modeling/) | Clustering and Topic Modeling | ⏳ Pending |
| [Chapter 6](./chapter-06-prompt-engineering/) | Prompt Engineering | ⏳ Pending |
| [Chapter 7](./chapter-07-advanced-generation/) | Advanced Text Generation | ⏳ Pending |
| [Chapter 8](./chapter-08-semantic-search-rag/) | Semantic Search & RAG | ⏳ Pending |
| [Chapter 9](./chapter-09-multimodal-llms/) | Multimodal LLMs | ⏳ Pending |

### Part III: Training and Fine-Tuning Models

| Chapter | Topic | Status |
|---------|-------|--------|
| [Chapter 10](./chapter-10-embedding-models/) | Creating Text Embedding Models | ⏳ Pending |
| [Chapter 11](./chapter-11-fine-tuning-classification/) | Fine-Tuning Classification Models | ⏳ Pending |
| [Chapter 12](./chapter-12-fine-tuning-generation/) | Fine-Tuning Generation Models | ⏳ Pending |

## 🔧 Common Issues

### transformers Version Issues

Some examples require specific versions of transformers. Check each chapter's `requirements.txt` file.

**Common version requirements**:
- Chapter 1 (Phi-3): `transformers==4.41.0`

### Device Compatibility

The book examples default to NVIDIA GPUs (`cuda`). On Apple Silicon, you need to change to `device_map="mps"`.

See each chapter's README for Apple Silicon adaptation instructions.

## 📝 Learning Notes

Each chapter directory contains:
- `README.md` - Chapter overview, learning objectives, personal notes
- Code examples - Python scripts
- `notebooks/` - Jupyter notebooks (if applicable)
- `outputs/` - Generated results
- `requirements.txt` - Chapter-specific dependencies

## 🔗 Resources

- [Book Website](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/)
- [Authors' GitHub](https://github.com/HandsOnLLM)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)

---

**Last Updated**: 2026-01-27
