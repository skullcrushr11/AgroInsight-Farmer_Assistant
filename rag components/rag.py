import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer


with open("paddy_knowledge_base.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)


index = faiss.read_index("paddy_vector_store_bge.index")


model = SentenceTransformer("BAAI/bge-large-en-v1.5")


def search_faiss(query, top_k=3):
    """
    Searches FAISS for the most relevant context based on the query.
    
    Args:
    - query (str): The input query/question.
    - top_k (int): Number of top results to return.

    Returns:
    - List of relevant text chunks.
    """
    
    query_embedding = model.encode([query], convert_to_numpy=True)

    
    distances, indices = index.search(query_embedding, top_k)

    
    results = [knowledge_base[i]["content"] for i in indices[0]]

    return results


import requests


LM_STUDIO_API_URL = "http://127.0.0.1:1234/v1/completions"


def ask_llm(query):
    retrieved_context = search_faiss(query)

    prompt = f"""You are an expert in paddy farming. 
    Use the following context to answer the query: 

    CONTEXT: {retrieved_context}

    QUESTION: {query}

    ANSWER:"""

    response = requests.post(
        LM_STUDIO_API_URL,
        json={"model": "mistral-7b-instruct-v0.3", "prompt": prompt, "max_tokens": 200}
    )
    

    return response


query = "What is the best water management practice for paddy?"
response = ask_llm(query)
print("LLM Response:", response)