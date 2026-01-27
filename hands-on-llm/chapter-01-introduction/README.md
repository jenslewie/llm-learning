# Chapter 1: Introduction to LLMs

> This chapter introduces the basic concepts of Large Language Models, including context windows, parameters, foundation models and other core concepts, with hands-on practice using the Phi-3-mini model.

## 📖 Learning Objectives

- Understand the evolution of LLMs
- Master key concepts: context window, parameters, foundation models
- Learn how to load and use pretrained models
- Understand the basic workflow of model inference

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Activate environment
conda activate llm-learning-phi-3-mini

# Install chapter dependencies (Important: must use specific version)
pip install transformers==4.41.0 torch accelerate
```

> [!WARNING]
> **Apple Silicon Users - Required Reading**
> - Must use `transformers==4.41.0` (do not use 5.0+)
> - Set `device_map` to `"mps"` in code
> 
> See: [Apple Silicon Setup Guide](../../docs/setup/apple-silicon-setup.md)

### 2. Run Example

```bash
python "Chapter 1 - Introduction to Language Models.py"
```

## 📝 Example Description

### `Chapter 1 - Introduction to Language Models.py`

Uses the Microsoft Phi-3-mini-4k-instruct model for text generation.

**Model Information**:
- Name: microsoft/Phi-3-mini-4k-instruct
- Parameters: ~3.8B
- Context Window: 4k tokens
- Model Size: ~7-8GB

**Key Code**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model (Apple Silicon adapted version)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="mps",              # ← Apple Silicon: use MPS
    torch_dtype="auto",
    trust_remote_code=True,
)

# Create generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
)

# Generate text
messages = [{"role": "user", "content": "Your prompt here"}]
output = generator(messages)
```

## 💡 Key Concepts Notes

### Context Window
- Phi-3-mini: 4096 tokens
- Determines how much context the model can "see"
- Content beyond the window will be truncated

### Parameters
- Phi-3-mini: ~3.8B parameters
- More parameters = stronger capabilities, but higher resource consumption

### Foundation Models
- Phi-3 is Microsoft's small foundation model
- Pretrained on large-scale text data
- Can be used directly or further fine-tuned

### Device Mapping
- `device_map="cuda"`: NVIDIA GPU
- `device_map="mps"`: Apple Silicon GPU
- `device_map="cpu"`: CPU (slow, not recommended)

## 🐛 Troubleshooting

### KeyError: 'type' in rope_scaling

**Cause**: transformers 5.0+ incompatible with Phi-3 model code

**Solution**:
```bash
pip install transformers==4.41.0
rm -rf ~/.cache/huggingface/modules/transformers_modules/microsoft/Phi*
```

### Slow Model Download

**For users in mainland China**:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### MPS Device Unavailable

**Requirements**: macOS ≥ 12.3

**Verify**:
```python
import torch
print(torch.backends.mps.is_available())  # Should return True
```

## 📊 Performance Reference

Tested on M4 Max:
- Model loading: ~4 seconds
- First inference: ~2-3 seconds
- Subsequent inference: ~1-2 seconds

## 🔗 Related Resources

- [Phi-3 Model Card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [Transformers Pipeline Documentation](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Apple MPS Backend Documentation](https://pytorch.org/docs/stable/notes/mps.html)

## ✅ Completion Checklist

After completing this chapter, you should be able to:

- [ ] Understand basic LLM concepts (context window, parameters, etc.)
- [ ] Load pretrained models using transformers
- [ ] Configure correct device mapping
- [ ] Use pipeline for text generation
- [ ] Troubleshoot common environment configuration issues

---

**Previous**: None  
**Next**: [Chapter 2 - Tokens and Embeddings](../chapter-02-tokens-embeddings/)
