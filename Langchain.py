import streamlit as st
import faiss
import json
import requests
import numpy as np
from langchain.llms.base import LLM
from typing import Optional, List
from sentence_transformers import CrossEncoder  # For reranking
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Streamlit UI Title
st.title("🌾 Paddy Farming AI Assistant (LangChain with Reranking)")

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

# Load reranker model (Cross-Encoder)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Wrap FAISS with LangChain retriever and reranking
def search_faiss_rerank(query, top_k=20, rerank_top_k=5):
    """Fetches top_k results from FAISS, reranks them, and returns the top rerank_top_k results."""
    
    # Convert query to embedding
    query_embedding = embedding_model.embed_query(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    
    # Retrieve documents from knowledge base
    retrieved_docs = [Document(page_content=knowledge_base[i]["content"]) for i in indices[0]]
    
    # Prepare input pairs for reranking (query + each document)
    doc_texts = [doc.page_content for doc in retrieved_docs]
    pairs = [[query, doc] for doc in doc_texts]
    
    # Compute rerank scores
    scores = reranker.predict(pairs)
    
    # Sort documents by rerank scores (higher score = more relevant)
    sorted_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
    
    # Return top rerank_top_k documents
    return sorted_docs[:rerank_top_k]

# LM Studio API details (Local Model)
LM_STUDIO_API_URL = "http://localhost:1234/v1/completions"

class LMStudioLLM(LLM):
    """Custom LangChain LLM wrapper for local LM Studio"""

    model_name: str = "falcon3-10b-instruct"  # Change if using a different model
    max_tokens: int = 1000
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

# Define the LangChain RetrievalQA Chain with Reranking
def get_langchain_answer(query):
    retrieved_docs = search_faiss_rerank(query)

    # Prepare context from reranked documents
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""You are an expert agricultural assistant with deep knowledge in paddy (rice) farming...  

*CONTEXT:*  
{context}  

*QUESTION:*  
{query}  

*Instructions:*  

1. Carefully review the provided context.  
2. Break down the question into relevant parts if needed.  
3. Base your answer solely on the context and reliable paddy farming practices.  
4. Provide a concise, accurate, and detailed answer that addresses the question fully.  
5. Strictly print every new point in a new line.  
6. Strictly print only the final answer.  
7. Even if you are unsure, the user should not know that.  
8. Don't print the chain of thought on the screen.  
9. Do not print the same sentence twice.
10. Do not miss any points.  
11. Always Start with a heading with a small introduction.

*FINAL ANSWER FORMAT EXAMPLE:*  

Final Answer Example:
Heading: Use appropriate introduction to the answer

- Use high-quality seeds to ensure better yield.  
- Maintain a proper water level of 5-7 cm during the vegetative stage.  
- Apply nitrogen fertilizers in three split doses for optimal growth.  
- Use pest control measures such as neem oil or pheromone traps.  



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
