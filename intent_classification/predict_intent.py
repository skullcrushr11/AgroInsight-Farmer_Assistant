import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

def load_model():
    """Load the fine-tuned ELECTRA model with LoRA adapters for intent classification"""
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    model_path = os.path.join(current_dir, "intent_classifier_model")
    label_mapping_path = os.path.join(current_dir, "label_mapping.json")
    
    
    with open(label_mapping_path, "r") as f:
        intent_to_label = json.load(f)
    
    
    label_to_intent = {v: k for k, v in intent_to_label.items()}
    
    
    base_model_name = "google/electra-base-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(intent_to_label)
    )
    
    
    model = PeftModel.from_pretrained(model, model_path)
    
    
    model.eval()
    
    return model, tokenizer, label_to_intent

def predict_intent(text, model, tokenizer, label_to_intent):
    """Predict the intent of a given text"""
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    )
    
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    
    
    predicted_intent = label_to_intent[predicted_class_id]
    
    
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    confidence = probabilities[0][predicted_class_id].item()
    
    return predicted_intent, confidence

def main():
    
    model, tokenizer, label_to_intent = load_model()
    
    
    test_prompts = [
        "What's the best crop to plant in clay soil?",
        "How much wheat can I grow per acre in Maharashtra?",
        "Can you recommend a fertilizer for tomatoes?",
        "What's causing these yellow spots on my plant leaves?",
        "How should I prepare my field for the monsoon season?",
        "My soil is very sandy, what nutrients does it need?",
        "Can AI help me choose which crop to plant?"
    ]
    
    
    print("=" * 80)
    print("INTENT CLASSIFICATION RESULTS")
    print("=" * 80)
    
    for i, prompt in enumerate(test_prompts):
        intent, confidence = predict_intent(prompt, model, tokenizer, label_to_intent)
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Predicted Intent: {intent}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    
    print("\n" + "=" * 80)
    print("INTERACTIVE TESTING (Type 'quit' to exit)")
    print("=" * 80)
    
    while True:
        user_input = input("\nEnter your query: ")
        if user_input.lower() == 'quit':
            break
        
        intent, confidence = predict_intent(user_input, model, tokenizer, label_to_intent)
        print(f"Predicted Intent: {intent}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

if __name__ == "__main__":
    main() 