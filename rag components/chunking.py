import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

input_folder = "Text data"  
output_file = "paddy_knowledge_base.json"


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=125,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""],  
)

knowledge_base = []

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as file:
            source_name = filename.replace(".txt", "")  
            content = file.read()

            
            chunks = text_splitter.split_text(content)

            
            for chunk in chunks:
                knowledge_base.append({"source": source_name, "content": chunk})


with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(knowledge_base, json_file, indent=4)

print(f"Knowledge base created with {len(knowledge_base)} chunks!")
