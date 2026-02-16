from transformers import pipeline
import torch

# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Load small model (fast)
print("Loading model...")
generator = pipeline(
    "text-generation",
    model="gpt2",
    device=0 if torch.cuda.is_available() else -1  # 0=GPU, -1=CPU
)

# Generate text
prompt = "The future of AI is"
print(f"\nPrompt: {prompt}")
output = generator(prompt, max_length=50, num_return_sequences=1)
print(f"Output: {output[0]['generated_text']}")