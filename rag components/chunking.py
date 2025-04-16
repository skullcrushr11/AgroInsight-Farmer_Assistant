import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

input_folder = "Text data"  # Folder containing .txt files
output_file = "paddy_knowledge_base.json"

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=125,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""],  # Ensures context retention
)

knowledge_base = []

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as file:
            source_name = filename.replace(".txt", "")  # Use filename as source identifier
            content = file.read()

            # Apply RecursiveCharacterTextSplitter
            chunks = text_splitter.split_text(content)

            # Store chunks in knowledge base
            for chunk in chunks:
                knowledge_base.append({"source": source_name, "content": chunk})

# Save as JSON
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(knowledge_base, json_file, indent=4)

print(f"Knowledge base created with {len(knowledge_base)} chunks!")
