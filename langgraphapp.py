import streamlit as st
import pandas as pd
import pickle
import joblib
import json
import requests
from langchain_community.vectorstores import FAISS
import numpy as np
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from typing import Dict, Any
import logging
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
from googletrans import Translator
import asyncio
import whisper
import sounddevice as sd
import scipy.io.wavfile as wavfile
import subprocess
import sys
import time
from datetime import datetime, timedelta
from collections import Counter

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


translator = Translator()


logging.basicConfig(filename="debug.log", level=logging.DEBUG, filemode="a")
logger = logging.getLogger(__name__)


def get_location():
    """Get latitude/longitude from IP using ip-api.com."""
    try:
        response = requests.get("http://ip-api.com/json")
        data = response.json()
        if data["status"] == "success":
            return data["lat"], data["lon"], data["city"], data["country"]
        else:
            logger.error("Failed to get location from ip-api.com")
            return None, None, None, None
    except Exception as e:
        logger.error(f"Error getting location: {str(e)}")
        return None, None, None, None

def classify_condition(precip, cloud_cover, sunshine_hours):
    """Classify daily weather as rainy, cloudy, sunny, or other."""
    if precip > 0.5:
        return "Rainy"
    elif cloud_cover > 60:
        return "Cloudy"
    elif cloud_cover < 30 and sunshine_hours > 4:
        return "Sunny"
    else:
        return "Mixed"

def get_weather_data(lat, lon, start_date, end_date):
    """Fetch weather data from Open-Meteo API for a given date range."""
    if not lat or not lon:
        logger.error("Invalid latitude/longitude")
        return None
    
    try:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
            "&daily=temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,"
            "cloudcover_mean,sunshine_duration&timezone=auto"
        )
        logger.info(f"Fetching data for {start_date} to {end_date}...")
        response = requests.get(url)
        data = response.json()
        
        if "error" in data:
            logger.error(f"API Error: {data.get('reason', 'Unknown error')}")
            return None
        if "daily" not in data or not data["daily"].get("time"):
            logger.error("No valid weather data returned from Open-Meteo")
            return None
        
        return data["daily"]
    
    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        return None

def process_weather_data(daily):
    """Process daily weather data into required metrics."""
    if not daily:
        return None
    
    monthly_rainfall = {}
    monthly_conditions = {}
    valid_temps = []
    valid_humidity = []
    
    for date, temp, humidity, precip, cloud, sunshine in zip(
        daily["time"],
        daily["temperature_2m_mean"],
        daily["relative_humidity_2m_mean"],
        daily["precipitation_sum"],
        daily["cloudcover_mean"],
        daily["sunshine_duration"]
    ):
        if any(x is None for x in [temp, humidity, precip, cloud, sunshine]):
            continue
        month = date[:7]
        monthly_rainfall[month] = monthly_rainfall.get(month, 0) + precip
        sunshine_hours = sunshine / 3600
        condition = classify_condition(precip, cloud, sunshine_hours)
        if month not in monthly_conditions:
            monthly_conditions[month] = []
        monthly_conditions[month].append(condition)
        valid_temps.append(temp)
        valid_humidity.append(humidity)
    
    if not valid_temps:
        return None
    
    avg_temp = sum(valid_temps) / len(valid_temps)
    avg_humidity = sum(valid_humidity) / len(valid_humidity)
    total_rainfall = sum(monthly_rainfall.values())
    highest_rainfall = max(monthly_rainfall.values()) if monthly_rainfall else None
    
    all_conditions = []
    for conditions in monthly_conditions.values():
        all_conditions.extend(conditions)
    condition_counter = Counter(all_conditions)
    
    
    dominant_condition = max(condition_counter, key=condition_counter.get) if condition_counter else None
    
    
    if dominant_condition == "Mixed":
        
        del condition_counter["Mixed"]
        
        if condition_counter:
            dominant_condition = max(condition_counter, key=condition_counter.get)
        else:
            
            dominant_condition = "Cloudy"
    
    return {
        "monthly_rainfall_mm": {k: round(v, 2) for k, v in monthly_rainfall.items()},
        "highest_monthly_rainfall_mm": round(highest_rainfall, 2) if highest_rainfall else None,
        "avg_temperature_celsius": round(avg_temp, 2),
        "avg_relative_humidity_percent": round(avg_humidity, 2),
        "moisture": round(avg_humidity / 100, 4),
        "total_rainfall_mm": round(total_rainfall, 2),
        "dominant_condition": dominant_condition
    }

def get_date_ranges():
    """Calculate date ranges based on current month."""
    now = datetime.now()
    current_year = now.year
    current_month = now.month
    
    
    prev_year = current_year - 1
    start_month_next = now.replace(month=current_month, year=prev_year, day=1)
    end_month_next = start_month_next + timedelta(days=89)
    next_3_start = start_month_next.strftime("%Y-%m-%d")
    next_3_end = end_month_next.strftime("%Y-%m-%d")
    
    
    start_month_last = now.replace(month=current_month, year=prev_year, day=1) - timedelta(days=60)
    end_month_last = now.replace(month=current_month, year=prev_year, day=1) + timedelta(days=29)
    last_3_start = start_month_last.strftime("%Y-%m-%d")
    last_3_end = end_month_last.strftime("%Y-%m-%d")
    
    
    start_month_current = now.replace(day=1) - timedelta(days=120)
    end_month_current = now  
    current_4_start = start_month_current.strftime("%Y-%m-%d")
    current_4_end = end_month_current.strftime("%Y-%m-%d")
    
    
    start_month_fallback = start_month_current.replace(year=prev_year)
    end_month_fallback = end_month_current.replace(year=prev_year)
    fallback_4_start = start_month_fallback.strftime("%Y-%m-%d")
    fallback_4_end = end_month_fallback.strftime("%Y-%m-%d")
    
    return {
        "prev_next_3": (next_3_start, next_3_end),
        "prev_last_3": (last_3_start, last_3_end),
        "current_last_4": (current_4_start, current_4_end),
        "fallback_last_4": (fallback_4_start, fallback_4_end)
    }


def check_ffmpeg():
    """Verify ffmpeg is installed and accessible."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        logger.info("ffmpeg is installed.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg not found")
        st.error("ERROR: ffmpeg not found. Please install it:\n- Run 'choco install ffmpeg' (with Chocolatey)\n- Or download from https://ffmpeg.org and add to PATH")
        return False

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone."""
    logger.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    logger.info("Recording finished.")
    return audio, sample_rate

def save_audio(audio, sample_rate):
    """Save audio to a WAV file in the current directory."""
    output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    audio_file = os.path.join(output_dir, "recorded_audio.wav")
    
    try:
        wavfile.write(audio_file, sample_rate, audio)
        logger.info(f"Audio saved to: {audio_file}")
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Failed to save audio file: {audio_file}")
        return audio_file
    except Exception as e:
        logger.error(f"Error saving audio: {str(e)}")
        raise

def transcribe_audio(audio_file, model_name="medium", language="en"):
    """Transcribe audio using Whisper with specified language."""
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found at {audio_file}")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        start_time = time.time()
        model = whisper.load_model(model_name).to(device)
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        start_time = time.time()
        result = model.transcribe(audio_file, language=language)
        transcribe_time = time.time() - start_time
        logger.info(f"Transcription took {transcribe_time:.2f} seconds")
        
        return {
            "text": result["text"],
            "language": language,
            "model_name": model_name
        }
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("CUDA out of memory! Trying 'base' model...")
            return transcribe_audio(audio_file, "base", language)
        else:
            logger.error(f"Error loading/transcribing: {str(e)}")
            return None
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@st.cache_resource
def load_models():
    try:
        
        with open("datasets/crop recommendation/crop_recommender_rf_model.pkl", "rb") as f:
            crop_model = pickle.load(f)
        with open("datasets/crop recommendation/label_encoder.pkl", "rb") as f:
            label_encoder_crop = pickle.load(f)
        
        
        yield_model = joblib.load("datasets/yield prediction/yield 1/random_forest_yield_model.pkl")
        label_encoder_yield = joblib.load("datasets/yield prediction/yield 1/label_encoders.pkl")
        
        
        fertilizer_model_1 = joblib.load("datasets/fertilizer prediction/fert_1/fertilizer_model_1.pkl")
        label_encoder_fert_1 = joblib.load("datasets/fertilizer prediction/fert_1/label_encoders_1.pkl")
        
        
        fertilizer_model_2 = joblib.load("datasets/fertilizer prediction/fert_2/fertilizer_model_2.pkl")
        label_encoder_fert_2 = joblib.load("datasets/fertilizer prediction/fert_2/label_encoders_2.pkl")
        remark_model_2 = joblib.load("datasets/fertilizer prediction/fert_2/remark_model_2.pkl")
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        disease_model = efficientnet_v2_s(weights=None)
        disease_model.classifier[1] = torch.nn.Linear(1280, 18)
        disease_model.load_state_dict(torch.load("disease_detection/best_model.pth", map_location=device))
        disease_model.eval().to(device)
        
        logger.info("Models loaded successfully")
        
        logger.debug(f"Type of yield_model: {type(yield_model)}")
        logger.debug(f"Type of label_encoder_yield: {type(label_encoder_yield)}")
        logger.debug(f"Type of fertilizer_model_1: {type(fertilizer_model_1)}")
        logger.debug(f"Type of label_encoder_fert_1: {type(label_encoder_fert_1)}")
        logger.debug(f"Type of fertilizer_model_2: {type(fertilizer_model_2)}")
        logger.debug(f"Type of label_encoder_fert_2: {type(label_encoder_fert_2)}")
        logger.debug(f"Type of remark_model_2: {type(remark_model_2)}")
        
        if isinstance(label_encoder_yield, dict):
            logger.debug(f"Keys in label_encoder_yield: {list(label_encoder_yield.keys())}")
        if isinstance(label_encoder_fert_1, dict):
            logger.debug(f"Keys in label_encoder_fert_1: {list(label_encoder_fert_1.keys())}")
        if isinstance(label_encoder_fert_2, dict):
            logger.debug(f"Keys in label_encoder_fert_2: {list(label_encoder_fert_2.keys())}")
            
        return (crop_model, label_encoder_crop, yield_model, label_encoder_yield,
                fertilizer_model_1, label_encoder_fert_1, fertilizer_model_2,
                label_encoder_fert_2, remark_model_2, disease_model)
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

(crop_model, label_encoder_crop, yield_model, label_encoder_yield,
 fertilizer_model_1, label_encoder_fert_1, fertilizer_model_2,
 label_encoder_fert_2, remark_model_2, disease_model) = load_models()


faiss_index_path = "vector_stores/unified_faiss_index"

@st.cache_resource
def load_vector_store():
    try:
        model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        index = FAISS.load_local(faiss_index_path, model, "unified_faiss_index", allow_dangerous_deserialization=True)
        logger.info("Vector store loaded successfully")
        return index, model
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise


faiss_index, embedding_model = load_vector_store()


LM_STUDIO_API_URL = "http://localhost:1234/v1/completions"
SMALL_MODEL = "falcon3-10b-instruct"
LARGE_MODEL = "falcon3-10b-instruct"


@st.cache_resource
def load_intent_classifier():
    """Load the fine-tuned DistilBERT model with LoRA adapters for intent classification"""
    try:
        
        INTENTS = [
            "General Farming Question",
            "Fertilizer Classification", 
            "Crop Recommendation",
            "Yield Prediction",
            "Image Plant Disease Detection",
            "Unclear"
        ]
        
        
        intent_dir = os.path.join(os.getcwd(), "intent_classification")
        model_path = os.path.join(intent_dir, "distilbert_lora_intent_classifier_final")
        
        
        id2label = {idx: intent for idx, intent in enumerate(INTENTS)}
        label2id = {intent: idx for idx, intent in enumerate(INTENTS)}
        
        
        base_model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(INTENTS),
            id2label=id2label,
            label2id=label2id
        )
        
        
        model = PeftModel.from_pretrained(model, model_path)
        
        
        model.eval()
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        logger.info(f"Intent classification model loaded successfully on {device}")
        return model, tokenizer, id2label, device
    except Exception as e:
        logger.error(f"Error loading intent classifier: {e}")
        return None, None, None, None


def reload_intent_classifier():
    """Reload the intent classifier model"""
    global intent_classifier
    st.cache_resource.clear(load_intent_classifier)
    intent_classifier = load_intent_classifier()
    return intent_classifier[0] is not None

def predict_intent(text):
    """Predict intent using the DistilBERT model"""
    model, tokenizer, id2label, device = intent_classifier
    
    if model is None:
        logger.error("Intent classification model not loaded, falling back to LLM")
        return query_llm(f"""Classify the user's intent into one of these categories:
        - Crop Recommendation
        - Yield Prediction
        - General Farming Question
        - Fertilizer Classification
        - Image Plant Disease Detection
        - Unclear

        User Message: "{text}"

        Return only the category name.
        """, model=SMALL_MODEL)
        
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    )
    
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    
    
    predicted_intent = id2label[predicted_class_id]
    
    
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    confidence = probabilities[0][predicted_class_id].item()
    
    logger.debug(f"Intent prediction: {predicted_intent} with confidence {confidence:.4f}")
    return predicted_intent


intent_classifier = None


disease_classes = [
    'cotton_bacterial_blight', 'cotton_curl_virus', 'cotton_fussarium_wilt', 'cotton_healthy',
    'maize_blight', 'maize_common_rust', 'maize_gray_leaf_spot', 'maize_healthy',
    'rice_bacterial_leaf_blight', 'rice_blast', 'rice_brown_spot', 'rice_healthy', 'rice_tungro',
    'wheat_brown_rust', 'wheat_fusarium_head_blight', 'wheat_healthy', 'wheat_mildew', 'wheat_septoria'
]


intent_classifier = load_intent_classifier()


def create_state():
    
    lat, lon, city, country = get_location()
    weather_data = {}
    
    if lat and lon:
        date_ranges = get_date_ranges()
        for period, (start_date, end_date) in date_ranges.items():
            daily = get_weather_data(lat, lon, start_date, end_date)
            weather_data[period] = process_weather_data(daily)
    
    return {
        "messages": [],
        "task": None,
        "user_input": None,
        "awaiting_input": False,
        "prediction": None,
        "last_user_message": None,
        "fertilizer_choice": None,
        "processed": False,
        "weather_data": weather_data,
        "location": {
            "lat": lat,
            "lon": lon,
            "city": city,
            "country": country
        }
    }


def search_faiss(query, metadata,top_k=10):
    try:
        results = faiss_index.similarity_search(
    query,
    k=10,
    filter={"crop": metadata}
    )
        return results
    except Exception as e:
        logger.error(f"Error in search_faiss: {e}")
        return []

def query_llm(prompt, model=SMALL_MODEL, is_formatting=False):
    try:
        logger.debug(f"Sending prompt to LLM ({model}): {prompt[:100]}...")
        response = requests.post(
            LM_STUDIO_API_URL,
            json={"model": model, "prompt": prompt, "max_tokens": 500, "stream": False},
            timeout=6000
        )
        response.raise_for_status()
        response_json = response.json()
        text = response_json.get("choices", [{}])[0].get("text", "").strip()
        if not text:
            raise ValueError("Empty response from LLM")
        text = text.replace("<|assistant|>", "").strip()
        
        
        if is_formatting:
            logger.debug(f"LLM ({model}) response (formatting): {text[:100]}...")
            return text
            
        
        logger.debug(f"LLM ({model}) response: {text[:100]}...")
        return text
    except Exception as e:
        logger.error(f"LLM API error ({model}): {e}")
        return f"Error: {str(e)}"

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(image).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return img

def process_general_question(query: str) -> str:
    prompt= f"""You are given a list of crop names and a query, and your task is to return a space-separated collection of crop names relevant to the query. 

*Crop List:*  
["wheat", "toordal", "rice", "ragi", "jowar", "groundnut", "cotton_hirustum", "cotton_arboreum", "corn", "coffee_arabica", "brinjal", "bengalgram", "bajra"]

*QUERY:*  
{query}

*Instructions:*

1. Review the provided Crop List and the Query carefully.  
2. Break down the Query into relevant parts if needed.  
3. If any crops from the list are relevant to the query, return them as a space-separated collection.
4. Crop names from the list are definitely relevant if they are present in the query.  
5. If no crops are directly relevant or if the query is general, return **all the crops** in the list, separated by a single space.  
6. Strictly follow this format: Return crop names with only a single space between them, no commas, no extra spaces at the beginning or end.  
7. Do not repeat any crop name in the list.  
8. If you're unsure of which crops are relevant, return all crops in the list, separated by a single space.  
9. Do not include anything other than the final space-separated answer.  
10. Do not provide any explanation, reasoning, or extra content.  
11. Do not provide any extra crop names if there are particular relevant crops to the query.

*FINAL ANSWER FORMAT EXAMPLE:*  

Final Answer Example:  
wheat jowar brinjal
"""
    english_response = query_llm(prompt, model=LARGE_MODEL, is_formatting=True)
    context=""
    for crop in english_response.split():
        context+=f"Context for {crop}:{search_faiss(query,metadata=crop)}\n"

    prompt = f"""You are an expert agricultural assistant with deep knowledge in farming...
CONTEXT:  
{context}  
QUESTION:  
{query}  
Instructions:  
1. Carefully review the provided context.  
2. Break down the question into relevant parts if needed.  
3. Base your answer solely on the context and reliable farming practices.  
4. Provide a concise, accurate, and detailed answer that addresses the question fully.  
5. Strictly print every new point in a new line.  
6. Strictly print only the final answer.  
7. Even if you are unsure, the user should not know that.  
8. Don't print the chain of thought on the screen.  
9. Do not print the same sentence twice.  
10. Do not miss any points.  
11. Always Start with a heading with a small introduction.
FINAL ANSWER:
    """
    english_response = query_llm(prompt, model=LARGE_MODEL, is_formatting=True)
    
    
    if st.session_state.selected_language != "en":
        try:
            translated_response = asyncio.run(translate_response_back(english_response, st.session_state.selected_language))
            return f"{translated_response}\n\n*Original English response:*\n{english_response}"
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return english_response
    else:
        return english_response

def input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Entering input_node")
    
    if not state["awaiting_input"] or not state["last_user_message"]:
        logger.debug("Not awaiting input or no user message, returning")
        return state
        
    user_message = state["last_user_message"]
    logger.debug(f"Processing user input: {user_message}")
    
    if state["task"] == "fertilizer_classification" and state["fertilizer_choice"] is None:
        if user_message.strip() in ["1", "2"]:
            state["fertilizer_choice"] = user_message.strip()
            state["awaiting_input"] = True  
            state["last_user_message"] = None
            logger.debug(f"Fertilizer choice set to: {state['fertilizer_choice']}")
            return state
        else:
            state["messages"].append({"role": "assistant", "content": "Please enter 1 or 2 to choose the fertilizer classification method."})
            state["last_user_message"] = None
            return state
    
    try:
        if state["task"] == "image_plant_disease_detection":
            logger.debug("Image input expected, handled by Streamlit UI")
        else:
            inputs = parse_user_input(user_message, state["task"], state.get("fertilizer_choice"), state.get("weather_data"))
            state["user_input"] = inputs
            state["awaiting_input"] = False
        state["last_user_message"] = None
        logger.debug(f"Parsed input: {state['user_input']}")
        return state
    except ValueError as e:
        state["messages"].append({"role": "assistant", "content": f"Sorry, I couldn't parse your input: {str(e)}. Please provide the values in the requested format."})
        state["last_user_message"] = None
        logger.debug(f"Input parsing failed: {e}")
        return state


def intent_classifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Entering intent_classifier_node")
    
    
    state["processed"] = False
    
    if not state["messages"]:
        state["messages"] = [{"role": "assistant", "content": "Hello! I can help with general questions, crop recommendations, yield predictions, fertilizer classification, or plant disease detection from images. What would you like to do?"}]
        return state

    
    if state["awaiting_input"] or (state["task"] and not state["processed"]):
        logger.debug("Skipping intent classification - task in progress or awaiting input")
        return state

    user_message = state["last_user_message"] if state["last_user_message"] else state["messages"][-1]["content"]
    logger.debug(f"User message for intent classification: {user_message}")

    
    intent = predict_intent(user_message)
    logger.debug(f"Detected intent: {intent}")

    if intent == "Unclear":
        state["messages"].append({"role": "assistant", "content": "I'm not sure what you want. Please clarify if you need a crop recommendation, yield prediction, general question, fertilizer classification, or plant disease detection from an image."})
        state["last_user_message"] = None
        state["task"] = None
        return state

    state["task"] = intent.lower().replace(" ", "_")
    state["last_user_message"] = None
    logger.debug(f"Task set to: {state['task']}")
    return state

def general_qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Entering general_qa_node")
    user_message = state["messages"][-1]["content"]
    response = process_general_question(user_message)
    state["messages"].append({"role": "assistant", "content": response})
    state["task"] = None
    state["processed"] = True
    logger.debug("Processed general question")
    return state

def crop_recommendation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Entering crop_recommendation_node")
    
    if state["processed"]:
        logger.debug("Crop recommendation already processed, returning")
        return state
        
    if not state["user_input"]:
        
        weather_info = ""
        if state["weather_data"] and "prev_next_3" in state["weather_data"] and state["weather_data"]["prev_next_3"]:
            weather = state["weather_data"]["prev_next_3"]
            weather_info = f"\nI have fetched the following weather data for your location:\n- Temperature: {weather['avg_temperature_celsius']}°C\n- Humidity: {weather['avg_relative_humidity_percent']}%\n- Rainfall: {weather['highest_monthly_rainfall_mm']} mm\n\n"
        
        state["messages"].append({"role": "assistant", "content": f"{weather_info}To recommend the best crop, please provide your approximate soil condions:\n- Nitrogen (N) in ppm\n- Phosphorus (P) in ppm\n- Potassium (K) in ppm\n- pH\nExample: 'N=90, P=42, K=43, ph=6.5'"})
        state["awaiting_input"] = True
        return state
        
    try:
        
        input_df = pd.DataFrame([{
            "N(ppm)": state["user_input"]["N(ppm)"],
            "P(ppm)": state["user_input"]["P(ppm)"],
            "K(ppm)": state["user_input"]["K(ppm)"],
            "temperature": state["user_input"]["temperature"],
            "humidity(relative humidity in %)": state["user_input"]["humidity(relative humidity in %)"],
            "ph": state["user_input"]["ph"],
            "rainfall(in mm)": state["user_input"]["rainfall(in mm)"]
        }])
        
        pred = crop_model.predict(input_df)
        crop = label_encoder_crop.inverse_transform(pred)[0]
        state["prediction"] = crop
        
        
        response = query_llm(f"""Format this prediction into a conversational response:
        The recommended crop based on the provided conditions is {crop}.
        Instructions:
        - Keep it friendly and concise.
        - Add a brief explanation or encouragement.""", model=SMALL_MODEL, is_formatting=True)
            
        state["messages"].append({"role": "assistant", "content": response})
        state["task"] = None
        state["user_input"] = None
        state["awaiting_input"] = False
        state["processed"] = True
        logger.debug(f"Crop predicted: {crop}")
    except Exception as e:
        error_msg = f"Error predicting crop: {str(e)}"
        state["messages"].append({"role": "assistant", "content": error_msg})
        state["task"] = None
        state["user_input"] = None
        state["awaiting_input"] = False
        state["processed"] = True
        logger.error(f"Crop prediction error: {e}")
    return state

def fertilizer_classification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Entering fertilizer_classification_node")
    
    if state["processed"]:
        logger.debug("Fertilizer classification already processed, returning")
        return state
        
    if state["fertilizer_choice"] is None:
        state["messages"].append({"role": "assistant", "content": "Please choose an option for fertilizer classification:\n1. Based on soil color\n2. Based on soil type \n\nEnter 1 or 2."})
        state["awaiting_input"] = True
        return state
    
    if state["fertilizer_choice"] == "1" and not state["user_input"]:
        
        weather_info = ""
        if state["weather_data"] and "prev_next_3" in state["weather_data"] and state["weather_data"]["prev_next_3"]:
            weather = state["weather_data"]["prev_next_3"]
            weather_info = f"\nI have fetched the following weather data for your location:\n- Temperature: {weather['avg_temperature_celsius']}°C\n\n"
        
        state["messages"].append({"role": "assistant", "content": f"{weather_info} For a fertilizer recommendation, Please provide the following soil conditions based on your farm:\n- Soil Color\n- Nitrogen (ppm)\n- Phosphorus (ppm)\n- Potassium (ppm)\n- pH\n- Crop\nExample: 'Soil_color=Dark Brown, Nitrogen=120, Phosphorous=80, Potassium=60, pH=6.5, Crop=Wheat'"})
        state["awaiting_input"] = True
        return state
        
    if state["fertilizer_choice"] == "2" and not state["user_input"]:
        
        weather_info = ""
        if state["weather_data"] and "prev_next_3" in state["weather_data"] and state["weather_data"]["prev_next_3"]:
            weather = state["weather_data"]["prev_next_3"]
            weather_info = f"\nI have fetched the following weather data for your location:\n- Temperature: {weather['avg_temperature_celsius']}°C\n- Moisture: {weather['moisture']}\n- Rainfall: {weather['highest_monthly_rainfall_mm']} mm\n\n"
        
        state["messages"].append({"role": "assistant", "content": f"{weather_info}For a fertilizer recommendation, Please provide the following soil conditions based on your farm:\n- pH\n- Nitrogen (ppm)\n- Phosphorus (ppm)\n- Potassium (ppm)\n- Carbon (%)\n- Soil Type (must be one of: Loamy Soil, Peaty Soil, Acidic Soil, Neutral Soil, Alkaline Soil)\n- Crop\nExample: 'ph=6.5, N=90, P=42, K=43, carbon=1.2, soil=Loamy Soil, crop=rice'."})
        state["awaiting_input"] = True
        return state

    try:
        input_dict = state["user_input"]
        logger.debug(f"Input dictionary: {input_dict}")
        
        
        input_df = pd.DataFrame([input_dict])
        
        if state["fertilizer_choice"] == "1":
            
            if not isinstance(label_encoder_fert_1, dict):
                raise TypeError("label_encoder_fert_1 is not a dictionary. Please check the saved model files.")
            
            
            categorical_cols = ["Soil_color", "Crop"]
            for col in categorical_cols:
                if col in input_df.columns and col in label_encoder_fert_1:
                    input_df[col] = label_encoder_fert_1[col].transform(input_df[col])
            
            logger.debug(f"Preprocessed input DataFrame: {input_df}")
            
            
            if not hasattr(fertilizer_model_1, 'predict'):
                raise TypeError("fertilizer_model_1 is not a valid model. Please check the saved model files.")
            
            
            pred = fertilizer_model_1.predict(input_df)
            
            
            if "Fertilizer" in label_encoder_fert_1:
                fertilizer = label_encoder_fert_1["Fertilizer"].inverse_transform(pred)[0]
            else:
                raise KeyError("Fertilizer key not found in label_encoder_fert_1")
            
            response = query_llm(f"""Format this prediction into a conversational response:
            The recommended fertilizer based on soil color is {fertilizer}.
            Instructions:
            - Keep it friendly and concise.
            - Add a brief encouragement.""", model=SMALL_MODEL, is_formatting=True)
            
        else:  
            
            if not isinstance(label_encoder_fert_2, dict):
                raise TypeError("label_encoder_fert_2 is not a dictionary. Please check the saved model files.")
            
            
            categorical_cols = ["Soil", "Crop"]
            for col in categorical_cols:
                if col in input_df.columns and col in label_encoder_fert_2:
                    input_df[col] = label_encoder_fert_2[col].transform(input_df[col])
            
            logger.debug(f"Preprocessed input DataFrame: {input_df}")
            
            
            if not hasattr(fertilizer_model_2, 'predict') or not hasattr(remark_model_2, 'predict'):
                raise TypeError("One or both of the fertilizer models are not valid. Please check the saved model files.")
            
            
            fertilizer_pred = fertilizer_model_2.predict(input_df)
            remark_pred = remark_model_2.predict(input_df)
            
            
            if "Fertilizer" in label_encoder_fert_2 and "Remark" in label_encoder_fert_2:
                fertilizer = label_encoder_fert_2["Fertilizer"].inverse_transform(fertilizer_pred)[0]
                remark = label_encoder_fert_2["Remark"].inverse_transform(remark_pred)[0]
            else:
                raise KeyError("Fertilizer or Remark key not found in label_encoder_fert_2")
            
            response = query_llm(f"""Format this prediction into a conversational response:
            The recommended fertilizer based on soil type is {fertilizer}.
            Additional remark: {remark}
            Instructions:
            - Keep it friendly and concise.
            - Add a brief encouragement.""", model=SMALL_MODEL, is_formatting=True)
        
        state["messages"].append({"role": "assistant", "content": response})
        state["task"] = None
        state["user_input"] = None
        state["fertilizer_choice"] = None
        state["awaiting_input"] = False
        state["processed"] = True
        logger.debug(f"Fertilizer prediction completed for choice {state['fertilizer_choice']}")
    except Exception as e:
        error_msg = str(e)
        if "previously unseen labels" in error_msg:
            error_msg = "Invalid soil type. Please use one of: Loamy Soil, Peaty Soil, Acidic Soil, Neutral Soil, Alkaline Soil"
        state["messages"].append({"role": "assistant", "content": f"Error: {error_msg}. Please try again with valid inputs."})
        state["user_input"] = None
        state["awaiting_input"] = True
        logger.error(f"Fertilizer prediction error: {e}")
    return state

def image_disease_detection_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Entering image_disease_detection_node")
    
    if state["processed"]:
        logger.debug("Image disease detection already processed, returning")
        return state
        
    if not state["user_input"]:
        state["messages"].append({"role": "assistant", "content": "Please upload an image of the plant to detect any diseases."})
        state["awaiting_input"] = True
        return state
        
    try:
        img = state["user_input"]  
        img_tensor = process_image(img)
        with torch.no_grad():
            outputs = disease_model(img_tensor)
            _, pred = torch.max(outputs, 1)
            disease = disease_classes[pred.item()]
        state["prediction"] = disease
        disease_info = process_general_question(f"What is {disease} and how to manage it?")
        response = query_llm(f"""Format this prediction into a conversational response:
        The detected plant disease is {disease}.
        Here's some info: {disease_info}
        Instructions:
        - Keep it friendly and concise.""", model=SMALL_MODEL, is_formatting=True)
        state["messages"].append({"role": "assistant", "content": response})
        state["task"] = None
        state["user_input"] = None
        state["awaiting_input"] = False
        state["processed"] = True
        logger.debug(f"Disease predicted: {disease}")
    except Exception as e:
        state["messages"].append({"role": "assistant", "content": f"Error detecting disease: {str(e)}"})
        state["task"] = None
        state["user_input"] = None
        state["awaiting_input"] = False
        state["processed"] = True
        logger.error(f"Disease detection error: {e}")
    return state

def yield_prediction_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Entering yield_prediction_node")
    
    if state["processed"]:
        logger.debug("Yield prediction already processed, returning")
        return state
        
    if not state["user_input"]:
        
        weather_info = ""
        if state["weather_data"] and "current_last_4" in state["weather_data"] and state["weather_data"]["current_last_4"]:
            weather = state["weather_data"]["current_last_4"]
            weather_info = f"\nI have fetched the following weather data for your location:\n- Temperature: {weather['avg_temperature_celsius']}°C\n- Rainfall: {weather['total_rainfall_mm']} mm\n- Weather Condition: {weather['dominant_condition']}\n\n"
        
        state["messages"].append({"role": "assistant", "content": f"{weather_info}To predict yield, please provide the following information:\n- Soil Type\n- Crop\n- Fertilizer Used (1 for Yes, 0 for No)\n- Irrigation Used (1 for Yes, 0 for No)\n- Days to Harvest\nExample: 'Soil_Type=Clay, Crop=Cotton, Fertilizer_Used=1, Irrigation_Used=1, Days_to_Harvest=120'"})
        state["awaiting_input"] = True
        return state
        
    try:
        input_dict = state["user_input"]
        logger.debug(f"Input dictionary: {input_dict}")
        
        
        input_df = pd.DataFrame([input_dict])
        
        
        categorical_cols = ["Soil_Type", "Crop", "Weather_Condition"]
        for col in categorical_cols:
            if col in input_df.columns and col in label_encoder_yield:
                input_df[col] = label_encoder_yield[col].transform(input_df[col])
        
        logger.debug(f"Preprocessed input DataFrame: {input_df}")
        
        
        if not hasattr(yield_model, 'predict'):
            raise TypeError("yield_model is not a valid model. Please check the saved model files.")
        
        
        pred = yield_model.predict(input_df)[0]
        state["prediction"] = f"{pred:.2f}"
        
        response = query_llm(f"""Format this prediction into a conversational response:
        The predicted yield based on the provided conditions is {pred:.2f} tons per hectare.
        Instructions:
        - Keep it friendly and concise.
        - Add a brief explanation or encouragement.""", model=SMALL_MODEL, is_formatting=True)
        
        state["messages"].append({"role": "assistant", "content": response})
        state["task"] = None
        state["user_input"] = None
        state["awaiting_input"] = False
        state["processed"] = True
        logger.debug(f"Yield predicted: {pred:.2f}")
    except Exception as e:
        state["messages"].append({"role": "assistant", "content": f"Error predicting yield: {str(e)}"})
        state["task"] = None
        state["user_input"] = None
        state["awaiting_input"] = False
        state["processed"] = True
        logger.error(f"Yield prediction error: {e}")
    return state

def parse_user_input(message: str, task: str, fertilizer_choice=None, weather_data=None) -> Dict[str, float]:
    logger.debug(f"Parsing input for task: {task}, fertilizer choice: {fertilizer_choice}")
    logger.debug(f"Input message: {message}")
    
    
    valid_soil_types = ['Loamy Soil', 'Peaty Soil', 'Acidic Soil', 'Neutral Soil', 'Alkaline Soil']
    valid_crops = ['rice', 'wheat', 'Mung Bean', 'Tea', 'millet', 'maize', 'Lentil', 'Jute', 
                  'Coffee', 'Cotton', 'Ground Nut', 'Peas', 'Rubber', 'Sugarcane', 'Tobacco',
                  'Kidney Beans', 'Moth Beans', 'Coconut', 'Black gram', 'Adzuki Beans',
                  'Pigeon Peas', 'Chickpea', 'banana', 'grapes', 'apple', 'mango', 'muskmelon',
                  'orange', 'papaya', 'pomegranate', 'watermelon']
    
    if task == "crop_recommendation":
        keys = ["N(ppm)", "P(ppm)", "K(ppm)", "temperature", "humidity(relative humidity in %)", "ph", "rainfall(in mm)"]
        key_aliases = {
            "n": "N(ppm)", "nitrogen": "N(ppm)",
            "p": "P(ppm)", "phosphorus": "P(ppm)",
            "k": "K(ppm)", "potassium": "K(ppm)",
            "temp": "temperature", "temperature": "temperature",
            "humidity": "humidity(relative humidity in %)",
            "ph": "ph", "pH": "ph",
            "rainfall": "rainfall(in mm)"
        }
        
        
        if weather_data and "prev_next_3" in weather_data and weather_data["prev_next_3"]:
            weather = weather_data["prev_next_3"]
            input_dict = {
                "temperature": weather["avg_temperature_celsius"],
                "humidity(relative humidity in %)": weather["avg_relative_humidity_percent"],
                "rainfall(in mm)": weather["highest_monthly_rainfall_mm"]
            }
        else:
            input_dict = {}
            
    elif task == "yield_prediction":
        
        keys = ["Soil_Type", "Crop", "Rainfall_mm", "Temperature_Celsius", "Fertilizer_Used", "Irrigation_Used", "Weather_Condition", "Days_to_Harvest"]
        key_aliases = {
            "soil": "Soil_Type", "soil_type": "Soil_Type",
            "crop": "Crop",
            "rainfall": "Rainfall_mm", "rain": "Rainfall_mm", "rainfall_mm": "Rainfall_mm",
            "temp": "Temperature_Celsius", "temperature": "Temperature_Celsius", "temperature_celsius": "Temperature_Celsius",
            "fertilizer": "Fertilizer_Used", "fertilizer_used": "Fertilizer_Used",
            "irrigation": "Irrigation_Used", "irrigation_used": "Irrigation_Used",
            "weather": "Weather_Condition", "weather_condition": "Weather_Condition",
            "days": "Days_to_Harvest", "days_to_harvest": "Days_to_Harvest"
        }
        
        
        if weather_data and "current_last_4" in weather_data and weather_data["current_last_4"]:
            weather = weather_data["current_last_4"]
            input_dict = {
                "Rainfall_mm": weather["total_rainfall_mm"],
                "Temperature_Celsius": weather["avg_temperature_celsius"],
                "Weather_Condition": weather["dominant_condition"]
            }
        else:
            input_dict = {}
            
    elif task == "fertilizer_classification":
        if fertilizer_choice == "1":
            
            keys = ["Soil_color", "Nitrogen", "Phosphorus", "Potassium", "pH", "Temperature", "Crop"]
            key_aliases = {
                "soil_color": "Soil_color", "soil": "Soil_color", "color": "Soil_color",
                "nitrogen": "Nitrogen", "n": "Nitrogen",
                "phosphorus": "Phosphorus", "phosphorous": "Phosphorus", "p": "Phosphorus",
                "potassium": "Potassium", "k": "Potassium",
                "ph": "pH", "pH": "pH",
                "temperature": "Temperature", "temp": "Temperature",
                "crop": "Crop"
            }
            
            
            if weather_data and "prev_next_3" in weather_data and weather_data["prev_next_3"]:
                weather = weather_data["prev_next_3"]
                input_dict = {
                    "Temperature": weather["avg_temperature_celsius"]
                }
            else:
                input_dict = {}
                
        elif fertilizer_choice == "2":
            keys = ["Temperature", "Moisture", "Rainfall", "PH", "Nitrogen", "Phosphorous", "Potassium", "Carbon", "Soil", "Crop"]
            key_aliases = {
                "temp": "Temperature", "temperature": "Temperature",
                "moisture": "Moisture",
                "rainfall": "Rainfall",
                "ph": "PH", "pH": "PH",
                "n": "Nitrogen", "nitrogen": "Nitrogen",
                "p": "Phosphorous", "phosphorous": "Phosphorous",
                "k": "Potassium", "potassium": "Potassium",
                "carbon": "Carbon",
                "soil": "Soil",
                "crop": "Crop"
            }
            
            
            if weather_data and "prev_next_3" in weather_data and weather_data["prev_next_3"]:
                weather = weather_data["prev_next_3"]
                input_dict = {
                    "Temperature": weather["avg_temperature_celsius"],
                    "Moisture": weather["moisture"],
                    "Rainfall": weather["highest_monthly_rainfall_mm"]
                }
            else:
                input_dict = {}
        else:
            raise ValueError("Fertilizer choice not set")
    else:
        raise ValueError("Invalid task")

    normalized_message = message.replace(" =", "=").replace("= ", "=").replace(" ,", ",").replace(", ", ",")
    pairs = [pair.strip() for pair in normalized_message.split(",")]
    logger.debug(f"Normalized pairs: {pairs}")
    
    for pair in pairs:
        if "=" in pair:
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            logger.debug(f"Processing pair - key: {key}, value: {value}")
            
            matched_key = None
            for alias, full_key in key_aliases.items():
                if key.lower() == alias.lower():
                    matched_key = full_key
                    break
            
            if matched_key:
                logger.debug(f"Matched key: {matched_key}")
                try:
                    
                    if task == "yield_prediction" and matched_key in ["Soil_Type", "Crop", "Weather_Condition"]:
                        input_dict[matched_key] = value
                        logger.debug(f"Added categorical value: {matched_key}={value}")
                    elif task == "fertilizer_classification" and fertilizer_choice == "2":
                        if matched_key == "Soil":
                            
                            if value not in valid_soil_types:
                                raise ValueError(f"Invalid soil type. Must be one of: {', '.join(valid_soil_types)}")
                            input_dict[matched_key] = value
                        elif matched_key == "Crop":
                            
                            if value not in valid_crops:
                                raise ValueError(f"Invalid crop. Must be one of: {', '.join(valid_crops)}")
                            input_dict[matched_key] = value
                        else:
                            input_dict[matched_key] = float(value)
                        logger.debug(f"Added value: {matched_key}={value}")
                    elif task == "fertilizer_classification" and matched_key in ["Soil_color", "Crop"]:
                        input_dict[matched_key] = value
                        logger.debug(f"Added categorical value: {matched_key}={value}")
                    else:
                        input_dict[matched_key] = float(value)
                        logger.debug(f"Added numeric value: {matched_key}={value}")
                except ValueError as e:
                    raise ValueError(f"Invalid value for {key}: {str(e)}")
            else:
                logger.debug(f"No match found for key: {key}")
                logger.debug(f"Available aliases: {list(key_aliases.keys())}")

    logger.debug(f"Final input dictionary: {input_dict}")
    missing_keys = set(keys) - set(input_dict.keys())
    if missing_keys:
        logger.debug(f"Missing keys: {missing_keys}")
        logger.debug(f"Available keys in input_dict: {list(input_dict.keys())}")
        raise ValueError(f"Missing inputs: {', '.join(missing_keys)}")
    
    
    if task == "yield_prediction":
        ordered_dict = {}
        for key in keys:
            if key in input_dict:
                ordered_dict[key] = input_dict[key]
        return ordered_dict
    
    
    if task == "fertilizer_classification" and fertilizer_choice == "1":
        ordered_dict = {}
        for key in keys:
            if key in input_dict:
                ordered_dict[key] = input_dict[key]
        return ordered_dict

    
    if task == "fertilizer_classification" and fertilizer_choice == "2":
        ordered_dict = {}
        for key in keys:
            if key in input_dict:
                ordered_dict[key] = input_dict[key]
        return ordered_dict
    
    return input_dict


workflow = StateGraph(dict)
workflow.add_node("intent_classifier", intent_classifier_node)
workflow.add_node("input", input_node)
workflow.add_node("general_qa", general_qa_node)
workflow.add_node("crop_recommendation", crop_recommendation_node)
workflow.add_node("fertilizer_classification", fertilizer_classification_node)
workflow.add_node("image_disease_detection", image_disease_detection_node)
workflow.add_node("yield_prediction", yield_prediction_node)

workflow.set_entry_point("intent_classifier")

workflow.add_conditional_edges(
    "intent_classifier",
    lambda state: (
        "input" if state["awaiting_input"] else 
        state["task"] if state["task"] and not state["processed"] else 
        END
    ),
    {
        "input": "input",
        "general_farming_question": "general_qa",
        "crop_recommendation": "crop_recommendation",
        "fertilizer_classification": "fertilizer_classification",
        "image_plant_disease_detection": "image_disease_detection",
        "yield_prediction": "yield_prediction",
        END: END
    }
)

workflow.add_conditional_edges(
    "input",
    lambda state: (
        state["task"] if state["task"] and (state["user_input"] or state["fertilizer_choice"]) and not state["processed"] else 
        END if state["processed"] else 
        "input"  
    ),
    {
        "general_farming_question": "general_qa",
        "crop_recommendation": "crop_recommendation",
        "fertilizer_classification": "fertilizer_classification",
        "image_plant_disease_detection": "image_disease_detection",
        "yield_prediction": "yield_prediction",
        "input": "input",
        END: END
    }
)

workflow.add_edge("general_qa", END)
workflow.add_edge("crop_recommendation", END)
workflow.add_edge("fertilizer_classification", END)
workflow.add_edge("image_disease_detection", END)
workflow.add_edge("yield_prediction", END)

app = workflow.compile()



async def detect_and_translate_to_english(user_input):
    original_lang = (await translator.detect(user_input)).lang
    if original_lang != "en":
        translated = await translator.translate(user_input, src=original_lang, dest='en')
        return translated.text, original_lang
    return user_input, original_lang

async def translate_response_back(response, target_lang):
    print("in translate \n")
    print("Response type:", type(response),"\n")
    print("Target language:", target_lang,"\n")
    if target_lang == 'en':
        return response
    translated = await translator.translate(response, src='en', dest=target_lang)
    print("in def trans:", translated)
    return translated.text


st.title("AgroInsight - Farming Assistant")


if "intent_classifier_tested" not in st.session_state:
    st.session_state.intent_classifier_tested = True
    with st.sidebar:
        st.subheader("Intent Classifier Status")
        try:
            
            test_query = "Tell me about paddy farming"
            intent = predict_intent(test_query)
            if intent in ["General Farming Question", "Crop Recommendation", "Yield Prediction", 
                         "Fertilizer Classification", "Image Plant Disease Detection", "Unclear"]:
                st.success(f"✅ Intent classifier is working (predicted '{intent}' for test query)")
            else:
                st.warning(f"⚠️ Intent classifier returned unexpected result: {intent}")
        except Exception as e:
            st.error(f"❌ Intent classifier error: {str(e)}")
            st.info("Using fallback LLM for intent classification")
            logger.error(f"Intent classifier test failed: {e}")


if "selected_language" not in st.session_state:
    st.session_state.selected_language = "en"


with st.sidebar:
    st.header("Language Settings")
    language_options = {
        "English": "en",
        "Hindi": "hi",
        "Telugu": "te",
        "Kannada": "kn",
        "Tamil": "ta",
        "Bengali": "bn",
        "Marathi": "mr",
        "Gujarati": "gu",
        "Punjabi": "pa"
    }
    selected_language_name = st.selectbox(
        "Select your preferred language",
        options=list(language_options.keys()),
        index=list(language_options.values()).index(st.session_state.selected_language)
    )
    st.session_state.selected_language = language_options[selected_language_name]
    
    
    st.header("Model Management")
    
    
    if st.button("Reload Intent Classifier"):
        try:
            if reload_intent_classifier():
                st.success("Intent classifier reloaded successfully!")
            else:
                st.error("Failed to reload intent classifier")
        except Exception as e:
            st.error(f"Error reloading intent classifier: {str(e)}")


if st.button("Clear Model Cache"):
    st.cache_resource.clear()
    st.success("Model cache cleared! Please wait for models to reload on the next action.")


if "chat_state" not in st.session_state:
    st.session_state.chat_state = create_state()


for msg in st.session_state.chat_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


if st.button("🎤 Speak your query"):
    if not check_ffmpeg():
        st.error("Please install ffmpeg to use speech input.")
    else:
        st.write("🎤 Recording... Speak now...")
        try:
            audio, sample_rate = record_audio(duration=5)
            audio_file = save_audio(audio, sample_rate)
            
            st.write("🔄 Transcribing...")
            result = transcribe_audio(audio_file, language=st.session_state.selected_language)
            if result:
                st.session_state.spoken_input = result["text"]
            else:
                st.error("Failed to transcribe audio. Please try again.")
            
            
            if os.path.exists(audio_file):
                os.remove(audio_file)
        except Exception as e:
            st.error(f"Error during recording: {str(e)}")

prompt = None
if "spoken_input" in st.session_state:
    prompt = st.session_state.spoken_input
    st.session_state.spoken_input = None
    print("speech prompt=", prompt)


text_prompt = st.chat_input("Ask me anything about farming!")


prompt_to_process = prompt or text_prompt


if st.session_state.chat_state["task"] == "image_plant_disease_detection":
    st.info("Please upload an image of your plant to detect any diseases.")
    uploaded_file = st.file_uploader("Upload Plant Image", type=["jpg", "jpeg", "png"], key="image_uploader")

    if uploaded_file:
        image = Image.open(uploaded_file)
        current_state = st.session_state.chat_state.copy()
        current_state["user_input"] = image
        current_state["awaiting_input"] = False

        with st.chat_message("user"):
            st.image(image, caption="Uploaded Plant Image", use_column_width=True)

        with st.spinner("Analyzing image..."):
            try:
                updated_state = app.invoke(current_state)
                st.session_state.chat_state = updated_state

                for msg in st.session_state.chat_state["messages"][-1:]:
                    if msg["role"] == "assistant":
                        with st.chat_message("assistant"):
                            st.write(msg["content"])
            except Exception as e:
                logger.error(f"Workflow invocation error: {e}")
                with st.chat_message("assistant"):
                    st.write(f"Sorry, something went wrong: {str(e)}")
        st.rerun()


elif prompt_to_process:
    original_input = prompt_to_process

    
    async def process_prompt(prompt_arg):
        
        translated_prompt, user_lang = await detect_and_translate_to_english(prompt_arg)

        with st.chat_message("user"):
            st.write(original_input)
            
            if user_lang != "en" and not any(x in translated_prompt.lower() for x in ["n=", "p=", "k=", "ph=", "temperature=", "humidity=", "rainfall="]):
                st.markdown("*Translated to English:* " + translated_prompt)

        
        current_state = st.session_state.chat_state.copy()
        current_state["messages"].append({"role": "user", "content": translated_prompt})
        current_state["last_user_message"] = translated_prompt

        with st.spinner("Thinking..."):
            try:
                updated_state = app.invoke(current_state)
                st.session_state.chat_state = updated_state

                for msg in updated_state["messages"][-1:]:
                    if msg["role"] == "assistant":
                        assistant_reply = msg["content"]

                        
                        if current_state["task"] == "crop_recommendation":
                            with st.chat_message("assistant"):
                                st.write(assistant_reply)
                        else:
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            with st.chat_message("assistant"):
                                st.write(assistant_reply)

            except Exception as e:
                logger.error(f"Workflow invocation error: {e}")
                with st.chat_message("assistant"):
                    st.write(f"Sorry, something went wrong: {str(e)}")

    
    asyncio.run(process_prompt(prompt_to_process))


if st.checkbox("Show debug logs"):
    if os.path.exists("debug.log"):
        with open("debug.log", "r") as f:
            st.text(f.read())
    else:
        st.text("No debug logs available yet.")