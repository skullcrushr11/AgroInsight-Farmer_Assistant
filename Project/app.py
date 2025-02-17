import streamlit as st
import faiss
import numpy as np
import json
import requests
from sentence_transformers import SentenceTransformer

# Load FAISS vector database
faiss_index_path = "paddy_vector_store_bge.index"
knowledge_base_path = "paddy_knowledge_base.json"

st.title("🌾 Paddy Farming AI Assistant")

# Load knowledge base
with open(knowledge_base_path, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# Load FAISS index
index = faiss.read_index(faiss_index_path)

# Load Sentence Transformer for embedding queries
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# LM Studio API details
LM_STUDIO_API_URL = "http://localhost:1234/v1/completions"
LLM_MODEL_NAME ="falcon3-7b-instruct"

# Function to retrieve relevant context from FAISS
def search_faiss(query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = [knowledge_base[i]["content"] for i in indices[0]]
    return results

# Function to query Mistral 7B via LM Studio API
def ask_llm(query):
    retrieved_context = search_faiss(query)

    prompt = f"""You are an expert agricultural assistant with deep knowledge in paddy (rice) farming. You have been trained on extensive, verified information regarding paddy cultivation—including soil requirements, water management, fertilizer and pesticide guidelines, pest control, sowing and harvesting schedules, and best agricultural practices.

    Below is some context extracted from a reliable paddy knowledge base. Use this context to answer the following question accurately and comprehensively. If the context is insufficient or if there are multiple aspects to the question, explain your reasoning step-by-step (Chain of Thought) before providing your final answer.

    CONTEXT:
    {retrieved_context}

    QUESTION:
    {query}

    Instructions:
    1. Carefully review the provided context.
    2. Break down the question into relevant parts if needed.
    3. Base your answer solely on the context and reliable paddy farming practices.
    4. Provide a concise, accurate, and detailed answer that addresses the question fully.
    5. Strictly print every new point in a new line(specially point 2).

    FINAL ANSWER:
    """

    response = requests.post(
        LM_STUDIO_API_URL,
        json={"model": LLM_MODEL_NAME, "prompt": prompt, "max_tokens": 300}
    )

    return response.json()["choices"][0]["text"]

# Streamlit UI
st.write("Ask any question related to **paddy farming**, and I will provide an expert response based on available knowledge.")

# User Input
user_query = st.text_input("🌾 Enter your question:", "")

if user_query:
    with st.spinner("Retrieving answer..."):
        response = ask_llm(user_query)
    
    st.subheader("📝 Answer:")
    st.write(response)
