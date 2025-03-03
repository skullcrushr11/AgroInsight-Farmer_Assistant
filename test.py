import requests

LM_STUDIO_API_URL = "http://localhost:1234/v1/completions"

test_prompt = "Explain why rice needs standing water for optimal growth."

response = requests.post(
    LM_STUDIO_API_URL,
    json={
        "model": "mistral-7b-instruct-v0.3",  # Ensure this matches the model loaded in LM Studio
        "prompt": test_prompt,
        "temperature": 0.7,
        "max_tokens": 200
    }
)

print(response.json())
