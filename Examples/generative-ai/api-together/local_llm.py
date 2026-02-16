import requests
from typing import Optional

class LocalLLM:
    """Wrapper to replace Together API calls"""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self.endpoint = f"{api_url}/api/generate"
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate text similar to together.Complete.create()"""
        
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                },
                timeout=120  # 2 minute timeout
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["text"]
            else:
                raise Exception(f"API Error: {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to local API. Is the server running?")

# Usage in your code
llm = LocalLLM()
output = llm.generate("What is machine learning?")
print(output)
