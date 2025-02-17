import streamlit as st
import faiss
import numpy as np
import json
import requests
import asyncio
import speech_recognition as sr
from googletrans import Translator
from sentence_transformers import SentenceTransformer

# Initialize speech recognizer and translator
recognizer = sr.Recognizer()
translator = Translator()

# Load FAISS vector database
faiss_index_path = "paddy_vector_store_bge.index"
knowledge_base_path = "paddy_knowledge_base.json"

st.title("🌾 Paddy Farming AI Chatbot")

# Load knowledge base
with open(knowledge_base_path, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# Load FAISS index
index = faiss.read_index(faiss_index_path)

# Load Sentence Transformer for embedding queries
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# LM Studio API details
LM_STUDIO_API_URL = "http://localhost:1234/v1/completions"
LLM_MODEL_NAME = "falcon-7b-instruct"

# 🔹 Function to retrieve relevant context from FAISS
def search_faiss(query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = [knowledge_base[i]["content"] for i in indices[0]]
    return results

# 🔹 Function to translate text using GoogleTrans
async def translate(text, dest_lang="en"):
    async with Translator() as translator:
        result = await translator.translate(text, dest=dest_lang)
        return result.text

# 🔹 Function to interact with Falcon LLM
def chat_with_llm(chat_history):
    """Formats chat history as a conversation and queries Falcon LLM."""
    formatted_chat = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history])

    prompt = f"""You are an expert agricultural assistant with deep knowledge in paddy (rice) farming.
    Maintain a friendly and helpful tone while responding to user queries.

    {formatted_chat}
    User: {chat_history[-1][0]}
    AI:"""

    response = requests.post(
        LM_STUDIO_API_URL,
        json={"model": LLM_MODEL_NAME, "prompt": prompt, "max_tokens": 300}
    )
    
    return response.json()["choices"][0]["text"]

# 🔹 Function to handle speech input
def recognize_speech():
    with sr.Microphone() as source:
        st.write("🎤 Speak now... (Listening...)")

        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.pause_threshold = 1.5
        recognizer.energy_threshold = 300

        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            text = recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return "⏳ No speech detected. Try again."
        except sr.UnknownValueError:
            return "🤷 Could not understand the speech."
        except sr.RequestError:
            return "❌ Speech recognition service unavailable."

# 🔹 Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🔹 Chat UI Layout: Chat history at the top, input bar at the bottom
st.markdown("<style>body{background-color: #f8f9fa;}</style>", unsafe_allow_html=True)

# Display chat history (newest messages at the bottom)
chat_container = st.container()
with chat_container:
    for user_text, ai_text in st.session_state.chat_history:
        st.markdown(f"**👤 You:** {user_text}", unsafe_allow_html=True)
        st.markdown(f"**🤖 AI:** {ai_text}", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

# Input bar at the bottom
st.markdown("---")
user_query = st.text_input("💡 Type your message:", "", key="user_input")

# 🎤 Speech input button
if st.button("🎙️ Speak"):
    spoken_query = recognize_speech()
    if spoken_query and "❌" not in spoken_query and "🤷" not in spoken_query:
        user_query = spoken_query  # Use speech input as the user query

if user_query:
    async def process_query():
        async with Translator() as translator:
            detected_lang = (await translator.detect(user_query)).lang
            
            if detected_lang != "en":
                user_query_translated = await translate(user_query, "en")
            else:
                user_query_translated = user_query

            with st.spinner("🤖 Generating response..."):
                st.session_state.chat_history.append((user_query_translated, ""))
                response = chat_with_llm(st.session_state.chat_history)
                st.session_state.chat_history[-1] = (user_query_translated, response)

            if detected_lang != "en":
                response = await translate(response, detected_lang)

            # Auto-scroll to latest message
            chat_container.empty()
            with chat_container:
                for user_text, ai_text in st.session_state.chat_history:
                    st.markdown(f"**👤 You:** {user_text}", unsafe_allow_html=True)
                    st.markdown(f"**🤖 AI:** {ai_text}", unsafe_allow_html=True)
                    st.markdown("<hr>", unsafe_allow_html=True)

    asyncio.run(process_query())
