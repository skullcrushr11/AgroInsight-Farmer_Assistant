import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import json
from pathlib import Path
from pprint import pprint
from huggingface_hub import login
login(token = "")

from langchain.text_splitter import RecursiveCharacterTextSplitter
#This code is for loading crop descriptions, text files, json files


model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

hf = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
def text_content(filepath):
    with open(filepath,'r') as f:
        return f.read()


L=["wheat","toordal","rice","ragi","jowar","groundnut","cotton_hirustum","cotton_arboreum","corn","coffee_arabica","brinjal","bengalgram","bajra"] 
#minilist=["wheat","toordal","rice"]
total_documents=[]
for i in L:
    print(i)
    path1="Crop_database\\"+i+".txt"
    content=text_content(path1)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 65, chunk_overlap=0)
    dlist = text_splitter.create_documents([content], metadatas=[{"crop": i}])


    path2="Crop_database\\"+i+".json"
    data = json.loads(Path(path2).read_text())
    content=json.dumps(data)
    document = Document(
        page_content=content,
        metadata={"crop": i}
    )
    dlist.append(document)
    total_documents+=dlist



index = faiss.IndexFlatL2(len(hf.embed_query("Could you tell me the optimal conditions for growing rice?")))
unified_vector_store= FAISS(
    embedding_function=hf,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
uuids = [str(uuid4()) for _ in range(len(total_documents))]
unified_vector_store.add_documents(documents=total_documents, ids=uuids)

unified_vector_store.save_local("vector_stores",f"unified_faiss_index")
