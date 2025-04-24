import json
with open("intent_dataset.json", "r") as f:
    data = json.load(f)
with open("intent_dataset.jsonl", "w") as f:
    for item in data:
        json.dump(item, f)
        f.write("\n")