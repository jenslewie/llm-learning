# Phi-3-mini Setup Guide for Apple Silicon (M4 Max)

This guide adapts the examples from Chapter 1 of "Hands-on Large Language Models" for macOS Apple Silicon environments.

## Environment Requirements

- **Operating System**: macOS
- **Chip**: Apple Silicon (M4 Max / M3 / M2 / M1 series)
- **Package Manager**: Conda/Miniconda

## 1. Create Conda Environment

```bash
# Create new conda environment
conda create -n llm-learning-phi-3-mini python=3.11 -y

# Activate environment
conda activate llm-learning-phi-3-mini
```

## 2. Install Dependencies

```bash
# Install PyTorch (with Apple Silicon MPS support)
pip install torch torchvision torchaudio

# Install transformers (Important: must use version 4.41.0)
pip install transformers==4.41.0

# Install other dependencies
pip install accelerate
```

### ⚠️ Version Compatibility

| Component | Recommended Version | Notes |
|-----------|---------------------|-------|
| `transformers` | **4.41.0** | ⚠️ Must use this version. Version 5.0+ is incompatible with current Phi-3 model code |
| `Python` | 3.11+ | Recommended to use 3.11 or newer |
| `torch` | Latest | Ensure MPS (Metal Performance Shaders) support |

## 3. Python Script

Create `Chapter 1 - Introduction to Language Models.py`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer (Apple Silicon adapted)
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="mps",              # Use MPS instead of CUDA
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Create text generation pipeline
print("Creating text generation pipeline...")
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)

# Example conversation
messages = [
    {"role": "user", "content": "Create a funny joke about chickens."}
]

# Generate output
print("Generating response...\n")
output = generator(messages)
print("Generated text:")
print(output[0]["generated_text"])
```

## 4. Run Script

```bash
# Ensure environment is activated
conda activate llm-learning-phi-3-mini

# Run script
python "Chapter 1 - Introduction to Language Models.py"
```

### First Run

On first run, the model will be automatically downloaded from Hugging Face Hub (~7-8GB), please be patient. The model is cached in `~/.cache/huggingface/` directory, and subsequent runs will use the cache directly.

## 📌 Important Notes

### 1. Device Mapping

> **⚠️ Critical Modification**
> 
> The book's original code uses `device_map="cuda"`, which is **not available** on macOS.
> 
> Apple Silicon must use: `device_map="mps"`

```python
# ❌ Book code (NVIDIA GPU only)
device_map="cuda"

# ✅ Apple Silicon adaptation
device_map="mps"
```

### 2. Transformers Version Lock

> **⚠️ Version Compatibility Issue**
> 
> Do not use `pip install transformers` to install the latest version!
> 
> transformers 5.0.0+ has RoPE configuration conflicts with current Phi-3 model code:
> - Error message: `KeyError: 'type'` in `modeling_phi3.py` line 296
> - Cause: Model code uses old `rope_scaling["type"]` format, new version uses `rope_parameters`

**Solution**: Lock to transformers 4.41.0

```bash
pip install transformers==4.41.0
```

### 3. Flash Attention Warnings

You may see the following warnings when running, which can be **safely ignored**:

```
`flash-attention` package not found
Current `flash-attention` does not support `window_size`
```

The model will automatically use the eager attention implementation, which still performs well on Apple Silicon.

### 4. Memory Requirements

- **Minimum Memory**: 16GB unified memory
- **Recommended Memory**: 32GB+ (M4 Max typically configured with 36GB/48GB)

## 🐛 Troubleshooting

### Issue 1: `KeyError: 'type'`

**Symptoms**:
```
KeyError: 'type'
scaling_type = self.config.rope_scaling["type"]
```

**Solution**:
```bash
# Downgrade transformers
pip install transformers==4.41.0

# Clear cache (if issue persists)
rm -rf ~/.cache/huggingface/modules/transformers_modules/microsoft/Phi*
```

### Issue 2: MPS Device Unavailable

**Symptoms**:
```
RuntimeError: MPS device not available
```

**Solution**:
- Ensure macOS version ≥ 12.3
- Update PyTorch to latest version: `pip install --upgrade torch`
- Verify MPS availability:
  ```python
  import torch
  print(torch.backends.mps.is_available())  # Should return True
  ```

### Issue 3: Slow Download Speed

**Symptoms**: Model download is very slow

**Solution**: Configure Hugging Face mirror (for users in mainland China)
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Or set in code:
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

## 📊 Performance Reference

Tested performance on M4 Max (Apple Silicon):

- **Model loading time**: ~4 seconds
- **First inference**: ~2-3 seconds
- **Subsequent inference**: ~1-2 seconds

## 🔗 Related Resources

- [Phi-3 Official Documentation](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch MPS Backend Documentation](https://pytorch.org/docs/stable/notes/mps.html)

## 🙋 Feedback

If you encounter issues, please check:
1. Is transformers version 4.41.0?
2. Is device_map set to "mps"?
3. Does PyTorch support MPS?

---

**Last Updated**: 2026-01-27  
**Test Environment**: macOS, M4 Max, Python 3.11, transformers 4.41.0
