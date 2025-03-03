import streamlit as st
import faiss
import json
import requests
import numpy as np
from langchain.llms.base import LLM
from typing import Optional, List
import requests
import json

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Streamlit UI Title
st.title("🌾 Paddy Farming AI Assistant (LangChain)")

# Paths for FAISS vector store and knowledge base
faiss_index_path = "paddy_vector_store_bge.index"
knowledge_base_path = "paddy_knowledge_base.json"

# Load knowledge base
with open(knowledge_base_path, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# Load Sentence Transformer for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Load FAISS index
index = faiss.read_index(faiss_index_path)

# Wrap FAISS with LangChain retriever
def search_faiss(query, top_k=5):
    query_embedding = embedding_model.embed_query(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    results = [Document(page_content=knowledge_base[i]["content"]) for i in indices[0]]
    return results

# LM Studio API details (Local Model)
LM_STUDIO_API_URL = "http://localhost:1234/v1/completions"

class LMStudioLLM(LLM):
    """Custom LangChain LLM wrapper for local LM Studio"""

    # Define required properties
    model_name: str = "falcon3-10b-instruct"  # Change if using a different model
    max_tokens: int = 300
    temperature: float = 0.7
    LM_STUDIO_API_URL: str = "http://localhost:1234/v1/completions"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager=None):
        """Sends the prompt to LM Studio and returns the response"""
        
        response = requests.post(
            self.LM_STUDIO_API_URL,
            json={"model": self.model_name, "prompt": prompt, "max_tokens": self.max_tokens},
        )

        if response.status_code == 200:
            response_json = response.json()
            return response_json["choices"][0]["text"]
        else:
            return "Error: Could not generate response."

    @property
    def _identifying_params(self):
        """Returns identifying parameters of the model"""
        return {"model_name": self.model_name, "max_tokens": self.max_tokens}

    @property
    def _llm_type(self):
        return "custom_LM_Studio_LLM"


# Define the LangChain RetrievalQA Chain
def get_langchain_answer(query):
    retrieved_docs = search_faiss(query)

    prompt = f"""You are an expert agricultural assistant with deep knowledge in paddy (rice) farming...  

*CONTEXT:*  
{retrieved_docs}  

*QUESTION:*  
{query}  

*Instructions:*  
1. Carefully review the provided context.  
2. Provide a concise, accurate, and detailed answer.  
3. Format the response properly with a heading and bullet points.

*FINAL ANSWER FORMAT EXAMPLE:*  

**Best Practices for Paddy Farming**  
- Use high-quality seeds to ensure better yield.  
- Maintain a proper water level of 5-7 cm during the vegetative stage.  
- Apply nitrogen fertilizers in three split doses.  

*FINAL ANSWER:*"""

    llm = LMStudioLLM()
    return llm(prompt)

# Streamlit UI
st.write("Ask any question related to *paddy farming*, and I will provide an expert response based on available knowledge.")

# User Input
user_query = st.text_input("🌾 Enter your question:", "")

if user_query:
    with st.spinner("Retrieving answer..."):
        response_generator = get_langchain_answer(user_query)

        st.subheader("📝 Answer:")
        answer_container = st.empty()

        full_response = ""
        for chunk in response_generator:
            full_response += chunk
            answer_container.write(full_response)
