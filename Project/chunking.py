import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def chunk_text(text, chunk_size=300):
    """
    Splits text into chunks of approximately 'chunk_size' words.
    
    Args:
    - text (str): The input text to be chunked.
    - chunk_size (int): Maximum words per chunk.

    Returns:
    - List of text chunks.
    """
    sentences = sent_tokenize(text)  # Split into sentences
    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()  # Split sentence into words
        word_count += len(words)

        if word_count > chunk_size:
            chunks.append(" ".join(current_chunk))  # Store previous chunk
            current_chunk = []  # Start a new chunk
            word_count = len(words)  # Reset word count
        
        current_chunk.append(sentence)

    # Add last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

import os
import json

input_folder = "Text data"  # Folder containing .txt files
output_file = "paddy_knowledge_base.json"

knowledge_base = []

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as file:
            source_name = filename.replace(".txt", "")  # Use filename as source identifier
            content = file.read()

            # Apply Chunking
            chunks = chunk_text(content, chunk_size=400)

            # Store chunks in knowledge base
            for chunk in chunks:
                knowledge_base.append({"source": source_name, "content": chunk})

# Save as JSON
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(knowledge_base, json_file, indent=4)

print(f"Knowledge base created with {len(knowledge_base)} chunks!")
