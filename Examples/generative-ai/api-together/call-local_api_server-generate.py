import requests

body = {
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
}

response = requests.post(
    "http://localhost:5000/api/generate",
    json=body,
    headers={"Content-Type": "application/json"},
)

print(response.text)