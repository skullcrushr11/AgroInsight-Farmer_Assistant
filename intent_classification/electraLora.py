import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def load_dataset(file_path):
    """Load the JSON dataset and convert to pandas DataFrame"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]  # Load as JSONL for large datasets
        df = pd.DataFrame(data)
        df = df.rename(columns={'query': 'input', 'intent': 'output'})
        logger.info(f"Loaded dataset with {len(df)} examples")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def compute_metrics(eval_pred):
    """Compute metrics for evaluation including accuracy, precision, recall, and F1-score"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def train_intent_classifier():
    """Train the intent classifier using LoRA fine-tuning with ELECTRA"""
    try:
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load dataset
        dataset_path = os.path.join(current_dir, "intent_dataset.jsonl")
        df = load_dataset(dataset_path)
        
        # Define intents (same as before, assuming all 6 are present)
        intent_to_label = {
            "Crop Recommendation": 0,
            "Yield Prediction": 1,
            "General Farming Question": 2,
            "Fertilizer Classification": 3,
            "Image Plant Disease Detection": 4,
            "Unclear": 5
        }
        
        # Validate intents in dataset
        unique_intents = df["output"].unique()
        invalid_intents = [intent for intent in unique_intents if intent not in intent_to_label]
        if invalid_intents:
            raise ValueError(f"Invalid intents found in dataset: {invalid_intents}")
        
        # Prepare data
        texts = df["input"].tolist()
        labels = [intent_to_label[intent] for intent in df["output"].tolist()]
        
        # Split dataset (stratified to handle class imbalance)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
        logger.info(f"Training set: {len(train_texts)} examples, Validation set: {len(val_texts)} examples")
        
        # Initialize tokenizer and model
        model_name = "google/electra-base-discriminator"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(intent_to_label)
        )
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "output.dense"]
        )
        
        # Get PEFT model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Create datasets
        train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
        val_dataset = IntentDataset(val_texts, val_labels, tokenizer)
        
        # Training arguments with paths in current directory
        training_args = TrainingArguments(
            output_dir=os.path.join(current_dir, "intent_classifier_output"),
            learning_rate=2e-4,
            per_device_train_batch_size=16,  # Increased for larger dataset
            per_device_eval_batch_size=16,
            num_train_epochs=8,  # Reduced to avoid overfitting
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
            logging_steps=100,
            save_total_limit=2,  # Limit saved checkpoints
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model and tokenizer in current directory
        model_save_path = os.path.join(current_dir, "intent_classifier_model")
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        # Evaluate on validation set
        val_results = trainer.evaluate()
        logger.info(f"Validation results: {val_results}")
        
        # Save label mapping in current directory
        label_mapping_path = os.path.join(current_dir, "label_mapping.json")
        with open(label_mapping_path, "w") as f:
            json.dump(intent_to_label, f)
        
        logger.info("Intent classifier training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_intent_classifier()