import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset, Features, ClassLabel, Value
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import json

# Define intents and label mapping
INTENTS = [
    "General Farming Question",
    "Fertilizer Classification",
    "Crop Recommendation",
    "Yield Prediction",
    "Image Plant Disease Detection",
    "Unclear"
]
label2id = {intent: idx for idx, intent in enumerate(INTENTS)}
id2label = {idx: intent for intent, idx in label2id.items()}

# Define features for the dataset
features = Features({
    'query': Value('string'),
    'intent': ClassLabel(names=INTENTS)
})

# Load and preprocess the JSON file
def load_and_preprocess_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure each item has the correct structure and valid intent
    processed_data = []
    for item in data:
        if isinstance(item, dict) and 'query' in item and 'intent' in item:
            # Validate intent
            if item['intent'] in INTENTS:
                processed_data.append({
                    'query': str(item['query']),
                    'intent': str(item['intent'])
                })
            else:
                print(f"Warning: Skipping invalid intent '{item['intent']}' for query: {item['query']}")
    
    # Save processed data to a temporary file
    temp_file = 'processed_data.json'
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    return temp_file

# Load and split dataset
processed_file = load_and_preprocess_json("intent_dataset.json")
dataset = load_dataset("json", data_files=processed_file, features=features)["train"]

# Split dataset: 80% train, 10% validation, 10% test
train_test_split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="intent")
train_val_split = train_test_split["train"].train_test_split(test_size=0.125, seed=42, stratify_by_column="intent")

dataset_dict = {
    "train": train_val_split["train"],      # 80%
    "validation": train_val_split["test"],  # 10%
    "test": train_test_split["test"]        # 10%
}

# Clean up temporary file
try:
    os.remove(processed_file)
except:
    pass

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(INTENTS),
    id2label=id2label,
    label2id=label2id
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocess dataset
def preprocess_function(examples):
    # Tokenize queries
    tokenized = tokenizer(examples["query"], truncation=True, padding=True, max_length=128)
    
    # Debug: Print intent values
    print("Intent values:", examples["intent"])
    
    # Convert intents to label IDs using the ClassLabel feature
    tokenized["labels"] = examples["intent"]
    
    return tokenized

# Apply preprocessing
tokenized_dataset = dataset_dict["train"].map(preprocess_function, batched=True)
tokenized_dataset_val = dataset_dict["validation"].map(preprocess_function, batched=True)
tokenized_dataset_test = dataset_dict["test"].map(preprocess_function, batched=True)

# Remove unnecessary columns and set format
for split in [tokenized_dataset, tokenized_dataset_val, tokenized_dataset_test]:
    split = split.remove_columns(["query", "intent"])
    split.set_format("torch")

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define compute_metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./distilbert_intent_classifier",
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,  # Enable mixed precision for CUDA
    report_to="none"  # Disable wandb logging
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(tokenized_dataset_test)
print("Test Results:", test_results)

# Save the fine-tuned model and tokenizer
output_dir = "./distilbert_intent_classifier_final"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

# Optional: Test inference locally
def classify_query(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    return id2label[predicted_id]

# Example usage
sample_queries = [
    "Tell me about corn",
    "What fertilizer for wheat?",
    "Best crop for sandy soil?",
    "How much yield for rice?",
    "How to treat leaf spots on maize?",
            "What's the best crop to plant in clay soil?",
        "How much wheat can I grow per acre in Maharashtra?",
        "Can you recommend a fertilizer for tomatoes?",
        "What's causing these yellow spots on my plant leaves?",
        "How should I prepare my field for the monsoon season?",
        "My soil is very sandy, what nutrients does it need?",
        "Can AI help me choose which crop to plant?"
]
for query in sample_queries:
    intent = classify_query(query)
    print(f"Query: {query} -> Intent: {intent}")