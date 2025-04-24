import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, ClassLabel
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

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

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(INTENTS),
    id2label=id2label,
    label2id=label2id
)

# Apply LoRA configuration
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=32,  # Scaling factor
    target_modules=["q_lin", "v_lin"],  # DistilBERT attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"  # Sequence classification
)
model = get_peft_model(model, lora_config)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# Load and split dataset
dataset = load_dataset("json", data_files="intent_dataset.json")["train"]

# Convert intent column to ClassLabel
intent_feature = ClassLabel(names=INTENTS)
dataset = dataset.cast_column("intent", intent_feature)

# Split dataset: 80% train, 10% validation, 10% test
train_test_split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="intent")
train_val_split = train_test_split["train"].train_test_split(test_size=0.125, seed=42, stratify_by_column="intent")

dataset_dict = {
    "train": train_val_split["train"],      # 80%
    "validation": train_val_split["test"],  # 10%
    "test": train_test_split["test"]        # 10%
}

# Preprocess dataset
def preprocess_function(examples):
    tokenized = tokenizer(examples["query"], truncation=True, padding=True, max_length=128)
    tokenized["labels"] = examples["intent"]  # Intent is already numeric after ClassLabel conversion
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
    output_dir="./distilbert_lora_intent_classifier",
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    num_train_epochs=30,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,  # Mixed precision for CUDA
    report_to="none"
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

# Save LoRA adapters and tokenizer
output_dir = "./distilbert_lora_intent_classifier_final"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"LoRA adapters and tokenizer saved to {output_dir}")

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
    "Tell me about paddy farming",
    "What fertilizer for wheat?",
    "Best crop for sandy soil?",
    "How much yield for rice?",
    "How to treat leaf spots on maize?",
    "What is this?"
    "there are spots on my maize leaves what is it?",
]
for query in sample_queries:
    intent = classify_query(query)
    print(f"Query: {query} -> Intent: {intent}")