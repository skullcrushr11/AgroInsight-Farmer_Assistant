

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json


with open("paddy_knowledge_base.json", "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)


model = SentenceTransformer("BAAI/bge-large-en-v1.5")


texts = [entry["content"] for entry in knowledge_base]
embeddings = model.encode(texts, convert_to_numpy=True)


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


faiss.write_index(index, "paddy_vector_store_bge.index")
print(f"Vector database created with {len(texts)} chunks!")
































