import requests

# LM Studio API URL
LM_STUDIO_API_URL = "http://localhost:1234/v1/completions"

# Function to ask LLM
def ask_llm(query):
    retrieved_context = search_faiss(query)

    prompt = f"""You are an expert in paddy farming. 
    Use the following context to answer the query: 

    CONTEXT: {retrieved_context}

    QUESTION: {query}

    ANSWER:"""

    response = requests.post(
        LM_STUDIO_API_URL,
        json={"model": "mistral-7b", "prompt": prompt, "max_tokens": 200}
    )

    return response.json()["choices"][0]["text"]

# Example Query
query = "What is the best water management practice for paddy?"
response = ask_llm(query)
print("LLM Response:", response)
