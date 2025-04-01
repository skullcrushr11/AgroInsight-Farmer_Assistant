import streamlit as st
import pandas as pd
import pickle
import joblib
import json
import requests
import faiss
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

# Set up logging
logging.basicConfig(filename="debug.log", level=logging.DEBUG, filemode="a")
logger = logging.getLogger(__name__)

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        # Load crop recommendation models
        with open("datasets/crop recommendation/crop_recommender_rf_model.pkl", "rb") as f:
            crop_model = pickle.load(f)
        with open("datasets/crop recommendation/label_encoder.pkl", "rb") as f:
            label_encoder_crop = pickle.load(f)
        
        # Load yield prediction models using joblib
        yield_model = joblib.load("datasets/yield prediction/yield 1/random_forest_yield_model.pkl")
        label_encoder_yield = joblib.load("datasets/yield prediction/yield 1/label_encoders.pkl")
        
        # Load fertilizer prediction models (fert_1) using joblib
        fertilizer_model_1 = joblib.load("datasets/fertilizer prediction/fert_1/fertilizer_model_1.pkl")
        label_encoder_fert_1 = joblib.load("datasets/fertilizer prediction/fert_1/label_encoders_1.pkl")
        
        # Load fertilizer prediction models (fert_2) using joblib
        fertilizer_model_2 = joblib.load("datasets/fertilizer prediction/fert_2/fertilizer_model_2.pkl")
        label_encoder_fert_2 = joblib.load("datasets/fertilizer prediction/fert_2/label_encoders_2.pkl")
        remark_model_2 = joblib.load("datasets/fertilizer prediction/fert_2/remark_model_2.pkl")
        
        # Load disease detection model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        disease_model = efficientnet_v2_s(weights=None)
        disease_model.classifier[1] = torch.nn.Linear(1280, 18)
        disease_model.load_state_dict(torch.load("disease_detection/best_model.pth", map_location=device))
        disease_model.eval().to(device)
        
        logger.info("Models loaded successfully")
        # Verify types of models and encoders
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

# --- Load FAISS Vector Store and Knowledge Base ---
faiss_index_path = "paddy_vector_store_bge.index"
knowledge_base_path = "paddy_knowledge_base.json"

@st.cache_resource
def load_vector_store():
    try:
        index = faiss.read_index(faiss_index_path)
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)
        model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        logger.info("Vector store loaded successfully")
        return index, knowledge_base, model
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise

faiss_index, knowledge_base, embedding_model = load_vector_store()

# --- LM Studio API Details ---
LM_STUDIO_API_URL = "http://localhost:1234/v1/completions"
SMALL_MODEL = "falcon3-10b-instruct"
LARGE_MODEL = "falcon3-10b-instruct"

# --- Disease Detection Classes ---
disease_classes = [
    'cotton_bacterial_blight', 'cotton_curl_virus', 'cotton_fussarium_wilt', 'cotton_healthy',
    'maize_blight', 'maize_common_rust', 'maize_gray_leaf_spot', 'maize_healthy',
    'rice_bacterial_leaf_blight', 'rice_blast', 'rice_brown_spot', 'rice_healthy', 'rice_tungro',
    'wheat_brown_rust', 'wheat_fusarium_head_blight', 'wheat_healthy', 'wheat_mildew', 'wheat_septoria'
]

# --- LangGraph State ---
def create_state():
    return {
        "messages": [],
        "task": None,
        "user_input": None,
        "awaiting_input": False,
        "prediction": None,
        "last_user_message": None,
        "fertilizer_choice": None,  # For Fertilizer Classification sub-options
        "processed": False  # New flag to prevent loops
    }

# --- Helper Functions ---
def search_faiss(query, top_k=10):
    try:
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_embedding, top_k)
        results = [knowledge_base[i]["content"] for i in indices[0]]
        return results
    except Exception as e:
        logger.error(f"Error in search_faiss: {e}")
        return []

def query_llm(prompt, model=SMALL_MODEL, is_formatting=False):
    try:
        logger.debug(f"Sending prompt to LLM ({model}): {prompt[:100]}...")
        response = requests.post(
            LM_STUDIO_API_URL,
            json={"model": model, "prompt": prompt, "max_tokens": 300, "stream": False},
            timeout=60
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
        valid_states = ["Crop Recommendation", "Yield Prediction", "General Paddy Farming Question",
                       "Fertilizer Classification", "Image Plant Disease Detection", "Unclear"]
        for state in valid_states:
            if state in text:
                logger.debug(f"LLM ({model}) response: {text[:100]}... | Matched state: {state}")
                return state
        logger.debug(f"LLM ({model}) response: {text[:100]}... | No valid state found, defaulting to Unclear")
        return "Unclear"
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

# --- LangGraph Nodes ---
def intent_classifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Entering intent_classifier_node")
    
    # Reset processed flag for new intents
    state["processed"] = False
    
    if not state["messages"]:
        state["messages"] = [{"role": "assistant", "content": "Hello! I can help with general questions, crop recommendations, yield predictions, fertilizer classification, or plant disease detection from images. What would you like to do?"}]
        return state

    # Skip intent classification if we're awaiting input or task is ongoing
    if state["awaiting_input"] or (state["task"] and not state["processed"]):
        logger.debug("Skipping intent classification - task in progress or awaiting input")
        return state

    user_message = state["last_user_message"] if state["last_user_message"] else state["messages"][-1]["content"]
    logger.debug(f"User message for intent classification: {user_message}")

    intent_prompt = f"""Classify the user's intent into one of these categories based on their message:
    - Crop Recommendation
    - Yield Prediction
    - General Paddy Farming Question
    - Fertilizer Classification
    - Image Plant Disease Detection
    - Unclear

    User Message: "{user_message}"

    Return only the category name.
    """
    intent = query_llm(intent_prompt, model=SMALL_MODEL)
    logger.debug(f"Detected intent: {intent}")

    if "Error" in intent or intent == "Unclear":
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
        state["messages"].append({"role": "assistant", "content": "To recommend the best crop, please provide:\n- Nitrogen (N) in ppm\n- Phosphorus (P) in ppm\n- Potassium (K) in ppm\n- Temperature (°C)\n- Humidity (%)\n- pH\n- Rainfall (mm)\nExample: 'N=90, P=42, K=43, temp=20.88, humidity=82, ph=6.5, rainfall=202.94'"})
        state["awaiting_input"] = True
        return state
        
    try:
        input_df = pd.DataFrame([state["user_input"]])
        pred = crop_model.predict(input_df)
        crop = label_encoder_crop.inverse_transform(pred)[0]
        state["prediction"] = crop
        response = query_llm(f"""Format this prediction into a conversational response:
        The recommended crop based on the provided conditions is {crop}.
        *Instructions:*
        - Keep it friendly and concise.
        - Add a brief explanation or encouragement.""", model=SMALL_MODEL, is_formatting=True)
        state["messages"].append({"role": "assistant", "content": response})
        state["task"] = None
        state["user_input"] = None
        state["awaiting_input"] = False
        state["processed"] = True
        logger.debug(f"Crop predicted: {crop}")
    except Exception as e:
        state["messages"].append({"role": "assistant", "content": f"Error predicting crop: {str(e)}"})
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
        state["messages"].append({"role": "assistant", "content": "Please choose an option for fertilizer classification:\n1. Based on soil color\n2. Based on soil type\nEnter 1 or 2."})
        state["awaiting_input"] = True
        return state
    
    if state["fertilizer_choice"] == "1" and not state["user_input"]:
        state["messages"].append({"role": "assistant", "content": "Please provide:\n- Soil Color\n- Nitrogen (ppm)\n- Phosphorus (ppm)\n- Potassium (ppm)\n- pH\n- Temperature (°C)\n- Crop\nExample: 'Soil_color=Red, Nitrogen=90, Phosphorous=42, Potassium=43, pH=6.5, Temperature=25, Crop=Rice'"})
        state["awaiting_input"] = True
        return state
        
    if state["fertilizer_choice"] == "2" and not state["user_input"]:
        state["messages"].append({"role": "assistant", "content": "Please provide:\n- Temperature (°C)\n- Moisture (%)\n- Rainfall (mm)\n- pH\n- Nitrogen (ppm)\n- Phosphorus (ppm)\n- Potassium (ppm)\n- Carbon (%)\n- Soil Type\n- Crop\nExample: 'temp=25, moisture=60, rainfall=150, ph=6.5, N=90, P=42, K=43, carbon=1.2, soil=clay, crop=rice'."})
        state["awaiting_input"] = True
        return state

    try:
        input_dict = state["user_input"]
        logger.debug(f"Input dictionary: {input_dict}")
        
        # Create DataFrame with consistent column names
        input_df = pd.DataFrame([input_dict])
        
        if state["fertilizer_choice"] == "1":
            # Handle first fertilizer model (based on soil color)
            if not isinstance(label_encoder_fert_1, dict):
                raise TypeError("label_encoder_fert_1 is not a dictionary. Please check the saved model files.")
            
            # Encode categorical variables
            categorical_cols = ["Soil_color", "Crop"]
            for col in categorical_cols:
                if col in input_df.columns and col in label_encoder_fert_1:
                    input_df[col] = label_encoder_fert_1[col].transform(input_df[col])
            
            logger.debug(f"Preprocessed input DataFrame: {input_df}")
            
            # Ensure fertilizer_model_1 is a RandomForestClassifier
            if not hasattr(fertilizer_model_1, 'predict'):
                raise TypeError("fertilizer_model_1 is not a valid model. Please check the saved model files.")
            
            # Predict using fertilizer_model_1
            pred = fertilizer_model_1.predict(input_df)
            
            # Decode prediction
            if "Fertilizer" in label_encoder_fert_1:
                fertilizer = label_encoder_fert_1["Fertilizer"].inverse_transform(pred)[0]
            else:
                raise KeyError("Fertilizer key not found in label_encoder_fert_1")
            
            response = query_llm(f"""Format this prediction into a conversational response:
            The recommended fertilizer based on soil color is {fertilizer}.
            *Instructions:*
            - Keep it friendly and concise.
            - Add a brief encouragement.""", model=SMALL_MODEL, is_formatting=True)
            
        else:  # state["fertilizer_choice"] == "2"
            # Handle second fertilizer model (based on soil type)
            if not isinstance(label_encoder_fert_2, dict):
                raise TypeError("label_encoder_fert_2 is not a dictionary. Please check the saved model files.")
            
            # Encode categorical variables
            categorical_cols = ["Soil", "Crop"]
            for col in categorical_cols:
                if col in input_df.columns and col in label_encoder_fert_2:
                    input_df[col] = label_encoder_fert_2[col].transform(input_df[col])
            
            logger.debug(f"Preprocessed input DataFrame: {input_df}")
            
            # Ensure both models are valid
            if not hasattr(fertilizer_model_2, 'predict') or not hasattr(remark_model_2, 'predict'):
                raise TypeError("One or both of the fertilizer models are not valid. Please check the saved model files.")
            
            # Predict both fertilizer and remark
            fertilizer_pred = fertilizer_model_2.predict(input_df)
            remark_pred = remark_model_2.predict(input_df)
            
            # Decode predictions
            if "Fertilizer" in label_encoder_fert_2 and "Remark" in label_encoder_fert_2:
                fertilizer = label_encoder_fert_2["Fertilizer"].inverse_transform(fertilizer_pred)[0]
                remark = label_encoder_fert_2["Remark"].inverse_transform(remark_pred)[0]
            else:
                raise KeyError("Fertilizer or Remark key not found in label_encoder_fert_2")
            
            response = query_llm(f"""Format this prediction into a conversational response:
            The recommended fertilizer based on soil type is {fertilizer}.
            Additional remark: {remark}
            *Instructions:*
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
        state["messages"].append({"role": "assistant", "content": f"Error predicting fertilizer: {str(e)}"})
        state["task"] = None
        state["user_input"] = None
        state["fertilizer_choice"] = None
        state["awaiting_input"] = False
        state["processed"] = True
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
        img = state["user_input"]  # Expecting a PIL Image from Streamlit file uploader
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
        *Instructions:*
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
        state["messages"].append({"role": "assistant", "content": "To predict yield, please provide:\n- Soil Type\n- Crop\n- Rainfall (mm)\n- Temperature (°C)\n- Fertilizer Used (1 for Yes, 0 for No)\n- Irrigation Used (1 for Yes, 0 for No)\n- Weather Condition\n- Days to Harvest\nExample: 'Soil_Type=Clay, Crop=Cotton, Rainfall_mm=800, Temperature_Celsius=25, Fertilizer_Used=1, Irrigation_Used=1, Weather_Condition=Sunny, Days_to_Harvest=120'"})
        state["awaiting_input"] = True
        return state
        
    try:
        input_dict = state["user_input"]
        logger.debug(f"Input dictionary: {input_dict}")
        
        # Create DataFrame with consistent column names
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical variables
        categorical_cols = ["Soil_Type", "Crop", "Weather_Condition"]
        for col in categorical_cols:
            if col in input_df.columns and col in label_encoder_yield:
                input_df[col] = label_encoder_yield[col].transform(input_df[col])
        
        logger.debug(f"Preprocessed input DataFrame: {input_df}")
        
        # Ensure yield_model is valid
        if not hasattr(yield_model, 'predict'):
            raise TypeError("yield_model is not a valid model. Please check the saved model files.")
        
        # Make prediction
        pred = yield_model.predict(input_df)[0]
        state["prediction"] = f"{pred:.2f}"
        
        response = query_llm(f"""Format this prediction into a conversational response:
        The predicted yield based on the provided conditions is {pred:.2f} tons per hectare.
        *Instructions:*
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
            state["awaiting_input"] = True  # Continue awaiting input for next step
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
            inputs = parse_user_input(user_message, state["task"], state.get("fertilizer_choice"))
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

def parse_user_input(message: str, task: str, fertilizer_choice=None) -> Dict[str, float]:
    logger.debug(f"Parsing input for task: {task}, fertilizer choice: {fertilizer_choice}")
    logger.debug(f"Input message: {message}")
    
    if task == "crop_recommendation":
        keys = ["N(ppm)", "P(ppm)", "K(ppm)", "temperature", "humidity(relative humidity in %)", "ph", "rainfall(in mm)"]
        key_aliases = {"n": "N(ppm)", "p": "P(ppm)", "k": "K(ppm)", "temp": "temperature", "humidity": "humidity(relative humidity in %)", "ph": "ph", "rainfall": "rainfall(in mm)"}
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
        # Define categorical columns for yield prediction
        categorical_cols = ["Soil_Type", "Crop", "Weather_Condition"]
    elif task == "fertilizer_classification":
        if fertilizer_choice == "1":
            keys = ["Soil_color", "Nitrogen", "Phosphorus", "Potassium", "pH", "Temperature", "Crop"]
            key_aliases = {
                "Soil_color": "Soil_color", "soil": "Soil_color", "color": "Soil_color",
                "Nitrogen": "Nitrogen", "nitrogen": "Nitrogen",
                "Phosphorus": "Phosphorus", "phosphorus": "Phosphorus", "phosphorous": "Phosphorus",
                "Potassium": "Potassium", "potassium": "Potassium",
                "pH": "pH", "ph": "pH",
                "Temperature": "Temperature", "temperature": "Temperature", "temp": "Temperature",
                "Crop": "Crop", "crop": "Crop"
            }
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
        else:
            raise ValueError("Fertilizer choice not set")
    else:
        raise ValueError("Invalid task")

    input_dict = {}
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
                    # Handle categorical variables
                    if task == "yield_prediction" and matched_key in categorical_cols:
                        input_dict[matched_key] = value
                        logger.debug(f"Added categorical value: {matched_key}={value}")
                    elif task == "fertilizer_classification" and matched_key in ["Soil_color", "Crop", "Soil"]:
                        input_dict[matched_key] = value
                        logger.debug(f"Added categorical value: {matched_key}={value}")
                    else:
                        input_dict[matched_key] = float(value)
                        logger.debug(f"Added numeric value: {matched_key}={value}")
                except ValueError:
                    raise ValueError(f"Invalid value for {key}: {value}")
            else:
                logger.debug(f"No match found for key: {key}")
                logger.debug(f"Available aliases: {list(key_aliases.keys())}")

    logger.debug(f"Final input dictionary: {input_dict}")
    missing_keys = set(keys) - set(input_dict.keys())
    if missing_keys:
        logger.debug(f"Missing keys: {missing_keys}")
        logger.debug(f"Available keys in input_dict: {list(input_dict.keys())}")
        raise ValueError(f"Missing inputs: {', '.join(missing_keys)}")
    
    return input_dict

def process_general_question(query: str) -> str:
    retrieved_context = search_faiss(query)
    prompt = f"""You are an expert agricultural assistant with deep knowledge in farming...
*CONTEXT:*  
{retrieved_context}  
*QUESTION:*  
{query}  
*Instructions:*  
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
*FINAL ANSWER:*
    """
    return query_llm(prompt, model=LARGE_MODEL, is_formatting=True)

# --- Define LangGraph Workflow ---
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
        "general_paddy_farming_question": "general_qa",
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
        "input"  # Stay in input if still awaiting
    ),
    {
        "general_paddy_farming_question": "general_qa",
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

# --- Streamlit Chat UI ---
st.title("🌾 Enhanced Farming Chatbot")

# At the top of your Streamlit UI section, after imports
if st.button("Clear Model Cache"):
    st.cache_resource.clear()  # Clears all resources cached with @st.cache_resource
    st.success("Model cache cleared! Please wait for models to reload on the next action.")

# Initialize state
if "chat_state" not in st.session_state:
    st.session_state.chat_state = create_state()

# Display chat history
for msg in st.session_state.chat_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle user input
prompt = st.chat_input("Ask me anything about farming!")

# Show image uploader when appropriate
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
                # Display only the latest assistant response
                for msg in st.session_state.chat_state["messages"][-1:]:
                    if msg["role"] == "assistant":
                        with st.chat_message("assistant"):
                            st.write(msg["content"])
            except Exception as e:
                logger.error(f"Workflow invocation error: {e}")
                with st.chat_message("assistant"):
                    st.write(f"Sorry, something went wrong: {str(e)}")
        st.rerun()  # Rerun to clear the file uploader

elif prompt:
    with st.chat_message("user"):
        st.write(prompt)
    current_state = st.session_state.chat_state.copy()
    current_state["messages"].append({"role": "user", "content": prompt})
    current_state["last_user_message"] = prompt

    with st.spinner("Thinking..."):
        try:
            updated_state = app.invoke(current_state)
            st.session_state.chat_state = updated_state
            # Display only the latest assistant response
            for msg in st.session_state.chat_state["messages"][-1:]:
                if msg["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.write(msg["content"])
        except Exception as e:
            logger.error(f"Workflow invocation error: {e}")
            with st.chat_message("assistant"):
                st.write(f"Sorry, something went wrong: {str(e)}")

# Debugging output
if st.checkbox("Show debug logs"):
    if os.path.exists("debug.log"):
        with open("debug.log", "r") as f:
            st.text(f.read())
    else:
        st.text("No debug logs available yet.")