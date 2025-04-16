#BGE large

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Load knowledge base
with open("paddy_knowledge_base.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# Load BGE Large Embedding Model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Convert text chunks into embeddings
texts = [entry["content"] for entry in knowledge_base]
embeddings = model.encode(texts, convert_to_numpy=True)

# Store in FAISS (Vector Database)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save FAISS index
faiss.write_index(index, "paddy_vector_store_bge.index")
print(f"Vector database created with {len(texts)} chunks!")


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



