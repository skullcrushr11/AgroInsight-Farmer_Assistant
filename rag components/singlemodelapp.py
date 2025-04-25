import streamlit as st
import faiss
import numpy as np
import json
import requests
from sentence_transformers import SentenceTransformer


faiss_index_path = "paddy_vector_store_bge.index"
knowledge_base_path = "paddy_knowledge_base.json"

st.title("🌾 Paddy Farming AI Assistant")


with open(knowledge_base_path, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)


index = faiss.read_index(faiss_index_path)


model = SentenceTransformer("BAAI/bge-large-en-v1.5")


LM_STUDIO_API_URL = "http://localhost:1234/v1/completions"


def search_faiss(query, top_k=10):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = [knowledge_base[i]["content"] for i in indices[0]]
    return results


def ask_large_llm(query):
    """
    Generates an answer using Mistral 7B.
    """
    retrieved_context = search_faiss(query)

    prompt = f"""You are an expert agricultural assistant with deep knowledge in paddy (rice) farming...  

*CONTEXT:*  
{retrieved_context}  

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


 *FINAL ANSWER:*
    """

    response = requests.post(
        LM_STUDIO_API_URL,
        json={"model": "falcon3-10b-instruct", "prompt": prompt, "max_tokens": 300, "stream": True},
        stream=True,  
    )

    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                decoded_line = line.decode("utf-8").replace("data: ", "").strip()
                if decoded_line == "[DONE]":
                    break  

                parsed_json = json.loads(decoded_line)  
                text_chunk = parsed_json["choices"][0]["text"]  
                full_response += text_chunk
                yield text_chunk  
            except Exception as e:
                print("Error parsing streamed response:", e)
    
    
    
    


st.write("Ask any question related to *paddy farming*, and I will provide an expert response based on available knowledge.")


user_query = st.text_input("🌾 Enter your question:", "")

if user_query:
    with st.spinner("Retrieving answer..."):
        response_generator = ask_large_llm(user_query)  
    
        st.subheader("📝 Answer:")
        answer_container = st.empty()  

        full_response = ""
        for chunk in response_generator:
            full_response += chunk  
            answer_container.write(full_response)  