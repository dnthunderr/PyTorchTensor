from local_llm import LocalLLM

llm = LocalLLM(api_url="http://localhost:5000")

# Use it
response = llm.generate(
    prompt="Your prompt",
    max_tokens=100
)
print(response)