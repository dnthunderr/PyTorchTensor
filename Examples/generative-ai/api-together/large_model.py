from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "mistralai/Mistral-7B-v0.1"
# or: "meta-llama/Llama-2-7b-hf"
# or: "gpt2-medium"

print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision for memory efficiency
    device_map="auto"  # Automatically use GPU
)

# Generate
prompt = "Explain machine learning:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
