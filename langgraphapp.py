import streamlit as st
import pandas as pd
import pickle
import json
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from typing import Dict, Any

# --- Load Models ---
@st.cache_resource
def load_models():
    with open("crop_recommender_rf_model.pkl", "rb") as f:
        crop_model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("yield_predictor_rf_model.pkl", "rb") as f:
        yield_model = pickle.load(f)
    return crop_model, label_encoder, yield_model

crop_model, label_encoder, yield_model = load_models()

# --- Load FAISS Vector Store and Knowledge Base ---
faiss_index_path = "paddy_vector_store_bge.index"
knowledge_base_path = "paddy_knowledge_base.json"

@st.cache_resource
def load_vector_store():
    index = faiss.read_index(faiss_index_path)
    with open(knowledge_base_path, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return index, knowledge_base, model

faiss_index, knowledge_base, embedding_model = load_vector_store()

# --- LM Studio API Details ---
LM_STUDIO_API_URL = "http://localhost:1234/v1/completions"

# --- LangGraph State ---
class State(Dict[str, Any]):
    task: str = None
    user_input: Dict[str, float] = None
    question: str = None
    prediction: str = None
    answer: str = None

# --- LangGraph Nodes ---
def input_node(state: State) -> State:
    if state["task"] == "crop_recommendation":
        state["user_input"] = {
            "N(ppm)": st.session_state.get("N_crop", 0),
            "P(ppm)": st.session_state.get("P_crop", 0),
            "K(ppm)": st.session_state.get("K_crop", 0),
            "temperature": st.session_state.get("temp_crop", 0),
            "humidity(relative humidity in %)": st.session_state.get("humidity_crop", 0),
            "ph": st.session_state.get("ph_crop", 0),
            "rainfall(in mm)": st.session_state.get("rainfall_crop", 0)
        }
    elif state["task"] == "yield_prediction":
        state["user_input"] = {
            "Fertilizer": st.session_state.get("fertilizer_yield", 0),
            "temp": st.session_state.get("temp_yield", 0),
            "N(ppm)": st.session_state.get("N_yield", 0),
            "P(ppm)": st.session_state.get("P_yield", 0),
            "K(ppm)": st.session_state.get("K_yield", 0)
        }
    elif state["task"] == "general_question":
        state["question"] = st.session_state.get("general_question", "")
    return state

def crop_prediction_node(state: State) -> State:
    input_df = pd.DataFrame([state["user_input"]])
    pred = crop_model.predict(input_df)
    state["prediction"] = label_encoder.inverse_transform(pred)[0]
    return state

def yield_prediction_node(state: State) -> State:
    input_df = pd.DataFrame([state["user_input"]])
    pred = yield_model.predict(input_df)
    state["prediction"] = f"{pred[0]:.2f}"
    return state

def search_faiss(query, top_k=10):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)
    results = [knowledge_base[i]["content"] for i in indices[0]]
    return results

def ask_large_llm(query):
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
            decoded_line = line.decode("utf-8").replace("data: ", "").strip()
            if decoded_line == "[DONE]":
                break
            parsed_json = json.loads(decoded_line)
            text_chunk = parsed_json["choices"][0]["text"]
            full_response += text_chunk
    return full_response

def general_question_node(state: State) -> State:
    state["answer"] = ask_large_llm(state["question"])
    return state

# --- Define LangGraph Workflow ---
workflow = StateGraph(State)
workflow.add_node("input", input_node)
workflow.add_node("crop_prediction", crop_prediction_node)
workflow.add_node("yield_prediction", yield_prediction_node)
workflow.add_node("general_question", general_question_node)

workflow.set_entry_point("input")
workflow.add_conditional_edges(
    "input",
    lambda state: state["task"],
    {
        "crop_recommendation": "crop_prediction",
        "yield_prediction": "yield_prediction",
        "general_question": "general_question"
    }
)
workflow.add_edge("crop_prediction", END)
workflow.add_edge("yield_prediction", END)
workflow.add_edge("general_question", END)

app = workflow.compile()

# --- Streamlit UI ---
st.title("🌾 Paddy Farming AI Assistant")

# Task selection
task = st.selectbox("Choose a Task", ["Crop Recommendation", "Yield Prediction", "General Paddy Farming Question"])

if task == "Crop Recommendation":
    st.subheader("Enter Soil and Weather Conditions")
    with st.form("crop_form"):
        st.session_state["N_crop"] = st.number_input("Nitrogen (N) in ppm", min_value=0.0, value=90.0)
        st.session_state["P_crop"] = st.number_input("Phosphorus (P) in ppm", min_value=0.0, value=42.0)
        st.session_state["K_crop"] = st.number_input("Potassium (K) in ppm", min_value=0.0, value=43.0)
        st.session_state["temp_crop"] = st.number_input("Temperature (°C)", min_value=0.0, value=20.88)
        st.session_state["humidity_crop"] = st.number_input("Humidity (%)", min_value=0.0, value=82.0)
        st.session_state["ph_crop"] = st.number_input("pH", min_value=0.0, value=6.5)
        st.session_state["rainfall_crop"] = st.number_input("Rainfall (mm)", min_value=0.0, value=202.94)
        submitted = st.form_submit_button("Predict Crop")

    if submitted:
        state = State(task="crop_recommendation")
        result = app.invoke(state)
        st.success(f"Recommended Crop: {result['prediction']}")

elif task == "Yield Prediction":
    st.subheader("Enter Fertilizer and Soil Conditions")
    with st.form("yield_form"):
        st.session_state["fertilizer_yield"] = st.number_input("Fertilizer Amount", min_value=0.0, value=80.0)
        st.session_state["temp_yield"] = st.number_input("Temperature (°C)", min_value=0.0, value=28.0)
        st.session_state["N_yield"] = st.number_input("Nitrogen (N) in ppm", min_value=0.0, value=80.0)
        st.session_state["P_yield"] = st.number_input("Phosphorus (P) in ppm", min_value=0.0, value=24.0)
        st.session_state["K_yield"] = st.number_input("Potassium (K) in ppm", min_value=0.0, value=20.0)
        submitted = st.form_submit_button("Predict Yield")

    if submitted:
        state = State(task="yield_prediction")
        result = app.invoke(state)
        st.success(f"Predicted Yield: {result['prediction']} tons/ha")

elif task == "General Paddy Farming Question":
    st.subheader("Ask a Paddy Farming Question")
    with st.form("question_form"):
        st.session_state["general_question"] = st.text_input("Enter your question:", value="What is the best crop for high rainfall?")
        submitted = st.form_submit_button("Get Answer")

    if submitted:
        with st.spinner("Retrieving answer..."):
            state = State(task="general_question")
            result = app.invoke(state)
            st.subheader("📝 Answer:")
            st.write(result["answer"])

# --- Footer ---
st.markdown("Powered by LangGraph, Random Forest Models, and LM Studio")