import os
import re
import json
import nltk
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) - {"below", "or", "and"}

# Directory containing Notepad text files
DATA_DIR = "knowledge"  # Assuming your notepad files are in this folder
OUTPUT_FILE = "cleaned_articles2.json"  # Output file for cleaned and chunked data

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
    text = re.sub(r"[^a-zA-Z0-9.,!?%\[\]\(\)−<> \u2212]", "", text)  # Remove special characters
    text = text.replace("\u2212", "-")
    return text.strip()

# Function to process text: tokenization, stopword removal, lemmatization
def process_text(text):
    sentences = sent_tokenize(text)  # Split into sentences
    processed_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)  # Tokenize words
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]  # Lemmatization & stopword removal
        processed_sentences.append(" ".join(words))
    
    return " ".join(processed_sentences)

# Function to chunk text into smaller segments, preserving paragraph boundaries
def chunk_text(text, chunk_size=256):
    paragraphs = text.split('\n')  # Split text into paragraphs (assuming paragraphs are separated by new lines)
    chunks = []
    current_chunk = []
    current_chunk_word_count = 0
    
    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)  # Tokenize paragraph into sentences
        for sentence in sentences:
            words = word_tokenize(sentence)  # Tokenize each sentence into words
            sentence_word_count = len(words)
            
            # If adding the sentence exceeds the chunk size, start a new chunk
            if current_chunk_word_count + sentence_word_count > chunk_size:
                chunks.append(" ".join(current_chunk))  # Add the current chunk to the list
                current_chunk = words  # Start a new chunk with the current sentence
                current_chunk_word_count = sentence_word_count
            else:
                current_chunk.extend(words)  # Add the sentence to the current chunk
                current_chunk_word_count += sentence_word_count
        
        # After processing all sentences in the paragraph, we add the chunk if necessary
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []  # Reset current chunk for the next paragraph
    
    return chunks

# Function to extract NER and POS from text
def extract_ner_pos(text):
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]  # Extract named entities
    pos_tags = [{"word": token.text, "pos": token.pos_} for token in doc]  # Extract POS tags
    return entities, pos_tags

# Read and process all text files
processed_data = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        file_path = os.path.join(DATA_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            raw_text = file.read()

        cleaned_text = clean_text(raw_text)  # Text cleaning
        processed_text = process_text(cleaned_text)  # Tokenization & Lemmatization
        chunks = chunk_text(processed_text)  # Chunking for vectorization
        
        # Extract NER and POS tags
        entities, pos_tags = extract_ner_pos(cleaned_text)
        
        # Store processed data
        processed_data.append({
            "filename": filename,
            "chunks": chunks,
            "entities": entities,
            "pos_tags": pos_tags
        })

# Save processed data to JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=4)

print(f"✅ Preprocessing complete! Data saved in {OUTPUT_FILE}")
