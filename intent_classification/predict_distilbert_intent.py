import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Define intents for the DistilBERT model
INTENTS = [
    "General Farming Question",
    "Fertilizer Classification", 
    "Crop Recommendation",
    "Yield Prediction",
    "Image Plant Disease Detection",
    "Unclear"
]

def load_model():
    """Load the fine-tuned DistilBERT model with LoRA adapters for intent classification"""
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    model_path = os.path.join(current_dir, "distilbert_lora_intent_classifier_final")
    
    # Create label mappings
    id2label = {idx: intent for idx, intent in enumerate(INTENTS)}
    label2id = {intent: idx for idx, intent in enumerate(INTENTS)}
    
    # Load base model and tokenizer
    base_model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(INTENTS),
        id2label=id2label,
        label2id=label2id
    )
    
    # Load the PEFT/LoRA adapters
    model = PeftModel.from_pretrained(model, model_path)
    
    # Set model to evaluation mode
    model.eval()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, tokenizer, id2label, device

def predict_intent(text, model, tokenizer, id2label, device):
    """Predict the intent of a given text"""
    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    )
    
    # Move inputs to the device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    
    # Map class ID to intent label
    predicted_intent = id2label[predicted_class_id]
    
    # Get confidence scores (softmax of logits)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    confidence = probabilities[0][predicted_class_id].item()
    
    return predicted_intent, confidence

def main():
    # Load model, tokenizer, and mappings
    model, tokenizer, id2label, device = load_model()
    
    # Hardcoded test prompts (same as in DistilLora.py plus some additional ones)
    test_prompts = [
        "Tell me about paddy farming",
        "What fertilizer for wheat?",
        "Best crop for sandy soil?",
        "How much yield for rice?",
        "How to treat leaf spots on maize?",
        "What is this?",
        "There are spots on my maize leaves what is it?",
        "What's the best crop to plant in clay soil?",
        "My tomato plants have black patches on the leaves",
        "How can I improve the yield of my cotton crop?",
        "Does rice need a lot of potassium?",
        "Can I grow carrots in my backyard?",
        "How do I know if my soil is healthy?"
    ]
    
    # Process each prompt
    print("=" * 80)
    print("DISTILBERT LORA INTENT CLASSIFICATION RESULTS")
    print("=" * 80)
    
    for i, prompt in enumerate(test_prompts):
        intent, confidence = predict_intent(prompt, model, tokenizer, id2label, device)
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Predicted Intent: {intent}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    # Allow interactive testing
    print("\n" + "=" * 80)
    print("INTERACTIVE TESTING (Type 'quit' to exit)")
    print("=" * 80)
    
    while True:
        user_input = input("\nEnter your query: ")
        if user_input.lower() == 'quit':
            break
        
        intent, confidence = predict_intent(user_input, model, tokenizer, id2label, device)
        print(f"Predicted Intent: {intent}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

if __name__ == "__main__":
    main() 