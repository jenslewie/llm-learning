from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer for Apple Silicon (MPS)
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="mps",  # Use MPS for Apple Silicon instead of "cuda"
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Create a pipeline
print("Creating text generation pipeline...")
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)

# Example messages
messages = [
    {"role": "user", "content": "Create a funny joke about chickens."}
]

# Generate output
print("Generating response...\n")
output = generator(messages)
print("Generated text:")
print(output[0]["generated_text"])
