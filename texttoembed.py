# BGE large

# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import json

# # Load knowledge base
# with open("paddy_knowledge_base.json", "r", encoding="utf-8") as f:
#     knowledge_base = json.load(f)

# # Load BGE Large Embedding Model
# model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# # Convert text chunks into embeddings
# texts = [entry["content"] for entry in knowledge_base]
# embeddings = model.encode(texts, convert_to_numpy=True)

# # Store in FAISS (Vector Database)
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(np.array(embeddings))

# # Save FAISS index
# faiss.write_index(index, "paddy_vector_store_bge.index")
# print(f"Vector database created with {len(texts)} chunks!")


# Roberta


# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import json

# # Load knowledge base
# with open("paddy_knowledge_base.json", "r", encoding="utf-8") as f:
#     knowledge_base = json.load(f)

# # Load RoBERTa Model for embeddings
# model = SentenceTransformer("sentence-transformers/roberta-large-nli-stsb-mean-tokens")

# # Convert text chunks into embeddings
# texts = [entry["content"] for entry in knowledge_base]
# embeddings = model.encode(texts, convert_to_numpy=True)

# # Store in FAISS (Vector Database)
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(np.array(embeddings))

# # Save FAISS index
# faiss.write_index(index, "paddy_vector_store_roberta.index")
# print(f"Vector database created with {len(texts)} chunks using RoBERTa!")


# ELMo

import faiss
import numpy as np
import json
import torch
from allennlp.commands.elmo import ElmoEmbedder

# Load knowledge base
with open("paddy_knowledge_base.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# Initialize ELMo embedder (pretrained model)
elmo = ElmoEmbedder()

# Function to compute sentence embeddings using ELMo
def elmo_sentence_embedding(sentence):
    """
    Computes a single vector for the sentence by averaging word embeddings.
    """
    tokens = sentence.split()
    if not tokens:
        return np.zeros((1024,))  # ELMo output size is 1024, return zeros for empty text
    
    # Get ELMo embeddings for all words (returns 3 layers: we use the top layer [-1])
    elmo_vectors = elmo.embed_sentence(tokens)
    sentence_embedding = np.mean(elmo_vectors[-1], axis=0)  # Average over words
    return sentence_embedding

# Convert text chunks into ELMo embeddings
texts = [entry["content"] for entry in knowledge_base]
embeddings = np.array([elmo_sentence_embedding(text) for text in texts])

# Store in FAISS (Vector Database)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "paddy_vector_store_elmo.index")

print(f"Vector database created with {len(texts)} chunks using ELMo!")
