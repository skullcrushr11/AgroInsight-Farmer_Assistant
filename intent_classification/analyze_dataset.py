import json
from collections import Counter

# Load the dataset
with open('intent_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"\nTotal records: {len(data)}")

# Validation checks
valid_intents = {
    "General Farming Question",
    "Fertilizer Classification",
    "Crop Recommendation",
    "Yield Prediction",
    "Image Plant Disease Detection",
    "Unclear"
}

# Initialize counters and error tracking
intent_counts = Counter()
missing_intents = 0
invalid_intents = 0
problematic_records = []

# Analyze each record
for idx, item in enumerate(data):
    if 'intent' not in item:
        missing_intents += 1
        problematic_records.append(f"Record {idx}: Missing intent field")
        continue
        
    intent = item['intent']
    if not intent:  # Check for empty strings or None
        missing_intents += 1
        problematic_records.append(f"Record {idx}: Empty intent")
        continue
        
    if intent not in valid_intents:
        invalid_intents += 1
        problematic_records.append(f"Record {idx}: Invalid intent '{intent}'")
        continue
        
    intent_counts[intent] += 1

# Print analysis results
print("\nData Analysis:")
print("-" * 50)
print(f"Total records in dataset: {len(data)}")
print(f"Records with valid intents: {sum(intent_counts.values())}")
print(f"Records with missing intents: {missing_intents}")
print(f"Records with invalid intents: {invalid_intents}")

print("\nLabel Distribution:")
print("-" * 50)
for intent in valid_intents:
    print(f"{intent}: {intent_counts[intent]} examples")

print("\nFirst 10 problematic records:")
print("-" * 50)
for record in problematic_records[:10]:
    print(record)
