import pandas as pd
import joblib

# Hardcoded mappings (these values should match your trained encodings)
CROP_MAPPING = {
    'Maize': 0,
    'Rice': 1,
    'Sugarcane': 2,
    'Cotton': 3,
    'Tobacco': 4
}

SOIL_COLOR_MAPPING = {
    'Black': 0,
    'Brown': 1,
    'Red': 2
}

def predict_fertilizer_simple(crop, soil_color, temperature, humidity, 
                            moisture, nitrogen, potassium, phosphorus, ph):
    # Load the model
    model = joblib.load("fertilizer_model_1.pkl")
    
    # Create input array with encoded categorical values
    input_data = pd.DataFrame([{
        'Crop': CROP_MAPPING.get(crop, -1),
        'Soil_color': SOIL_COLOR_MAPPING.get(soil_color, -1),
        'Temperature': temperature,
        'Unnamed: 8': humidity,
        'Moisture_': moisture,
        'Nitrogen': nitrogen,
        'Potassium': potassium,
        'Phosphorus': phosphorus,
        'pH': ph
    }])
    
    # Make prediction
    prediction_encoded = model.predict(input_data)
    
    # Hardcoded fertilizer mapping (reverse of what model predicts)
    fertilizer_mapping = {
        0: 'Urea',
        1: 'DAP',
        2: 'NPK',
        3: '14-35-14',
        4: '28-28',
        5: '17-17-17',
        6: '20-20',
        7: '10-26-26'
    }
    
    return fertilizer_mapping.get(prediction_encoded[0], 'Unknown')

if __name__ == "__main__":
    # Test case 1
    result = predict_fertilizer_simple(
        crop='Maize',
        soil_color='Black',
        temperature=25,
        humidity=65,
        moisture=35,
        nitrogen=80,
        potassium=40,
        phosphorus=35,
        ph=6.5
    )
    print(f"Test 1 Recommended Fertilizer: {result}")
    
    # Test case 2
    result = predict_fertilizer_simple(
        crop='Rice',
        soil_color='Brown',
        temperature=28,
        humidity=70,
        moisture=40,
        nitrogen=70,
        potassium=45,
        phosphorus=40,
        ph=7.0
    )
    print(f"Test 2 Recommended Fertilizer: {result}")
