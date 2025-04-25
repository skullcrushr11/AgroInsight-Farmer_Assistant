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


def search_faiss(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = [knowledge_base[i]["content"] for i in indices[0]]
    return results


def ask_large_llm(query):
    """
    Generates an answer using Mistral 7B.
    """
    retrieved_context = search_faiss(query)

    prompt = f"""You are an expert agricultural assistant with deep knowledge in paddy (rice) farming.

    CONTEXT:
    {retrieved_context}

    QUESTION:
    {query}

    Instructions:
    1. Carefully review the provided context.
    2. Break down the question into relevant parts if needed.
    3. Base your answer solely on the context and reliable paddy farming practices.
    4. Provide a concise, accurate, and detailed answer that addresses the question fully.
    5. Strictly print every new point in a new line (especially point 2).

    FINAL ANSWER:
    """

    response = requests.post(
        LM_STUDIO_API_URL,
        json={"model": "meta-llama-3.1-8b-instruct", "prompt": prompt, "max_tokens": 150, "stream": True},
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
    
    
    formatted_response = format_with_small_llm(full_response)
    yield f"\n\n---\n\n{formatted_response}"  


def format_with_small_llm(text):
    """
    Uses a small LLM (Phi-2) to format and structure the output without changing content.
    """
    formatting_prompt = f"""
    You are a professional text formatter.  
    Your task is to **reformat the given input text** into a well-structured, readable, and properly indented format.  

    **Rules:**  
    1. **DO NOT change or alter any content** - only format it.  
    2. Use **proper headings** where needed.  
    3. **Ensure correct indentation** (especially for point 2).  
    4. **Use bullet points** or **numbered lists** where applicable.  
    5. **DO NOT add extra information or modify the meaning**.  

    
    **Before Formatting:**  
    ```
    Paddy requires controlled irrigation. 
    One should avoid over-watering. 
    Applying fertilizers at the correct time is important. 
    Also, pests such as brown planthoppers can be prevented using integrated pest management techniques.
    ```
    
    **After Formatting:**  
    
    🌾 Paddy Irrigation & Fertilization Guide  

    1 Water Management  
    - Maintain proper irrigation levels.  
    - Avoid over-watering to prevent root rot.  

    2 Fertilizer Application  
    - Apply fertilizers at the correct growth stages.  

    3 Pest Control  
    - Use **Integrated Pest Management (IPM)** to prevent brown planthoppers.  
    

    ---
    
    Now, reformat the following input text in the same structured way:  

    **Input Text:**  
    ```
    {text}
    ```

    **Reformatted Output:**
    End generating after reformatting the output. Do not include any of the instructions in the output, only the reformatted input.Strictly avoid anything enclosed between ** and **.
    Do not include anything that you talk to yourself or ask yourself.
    """

    response = requests.post(
        LM_STUDIO_API_URL,
        json={"model":"meta-llama-3.1-8b-instruct", "prompt": formatting_prompt, "max_tokens": 150}
    )

    return response.json()["choices"][0]["text"]


st.write("Ask any question related to **paddy farming**, and I will provide an expert response based on available knowledge.")


user_query = st.text_input("🌾 Enter your question:", "")

if user_query:
    with st.spinner("Retrieving answer..."):
        response_generator = ask_large_llm(user_query)  
    
        full_response = ""
        for chunk in response_generator:
            full_response += chunk  

    
    st.subheader("📝 Answer (Formatted):")
    st.write(full_response)

