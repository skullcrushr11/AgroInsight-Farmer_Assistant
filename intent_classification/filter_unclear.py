import json

# Input and output file paths
input_file = "agri_intent_dataset_final.jsonl"
output_file = "unclear_intents.json"

# Store filtered results 
unclear_queries = []

# Read and process the JSONL file
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data["output"] == "Unclear":
            # Transform to new format
            transformed = {
                "query": data["input"],
                "intent": "Unclear"
            }
            unclear_queries.append(transformed)

# Write to new JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(unclear_queries, f, indent=2)

print(f"Found {len(unclear_queries)} unclear queries and saved to {output_file}")
