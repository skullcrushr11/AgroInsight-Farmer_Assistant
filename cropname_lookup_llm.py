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
login(token = "HuggingFACE_API_KEY")

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

hf = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


vector_store_map={}
L=["wheat","toordal","rice","ragi","jowar","groundnut","cotton_hirustum","cotton_arboreum","corn","coffee_arabica","brinjal","bengalgram","bajra"] 
for i in L:
    vector_store_map[i]=FAISS.load_local(f"vector_stores/{i}_faiss_index",hf,f"index", allow_dangerous_deserialization=True)
results = vector_store_map['rice'].similarity_search(
    "Could you tell me the optimal conditions for growing rice?",
    k=10,
)
context=[]
for res in results:
    #print(f"* {res.page_content} [{res.metadata}]")
    context.append(res.page_content)


from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
repo_id = "tiiuae/Falcon3-10B-Instruct"
llm = HuggingFaceEndpoint(
    task="text-generation",
    repo_id=repo_id,
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)

user_prompt=input("Enter your Query:")
prompt= f"""You are given a list of crop names and a query, and your task is to return a space-separated collection of crop names relevant to the query. 

*Crop List:*  
["wheat", "toordal", "rice", "ragi", "jowar", "groundnut", "cotton_hirustum", "cotton_arboreum", "corn", "coffee_arabica", "brinjal", "bengalgram", "bajra"]

*QUERY:*  
{user_prompt}

*Instructions:*

1. Review the provided Crop List and the Query carefully.  
2. Break down the Query into relevant parts if needed.  
3. If any crops from the list are relevant to the query, return them as a space-separated collection.
4. Crop names from the list are definitely relevant if they are present in the query.  
5. If no crops are directly relevant or if the query is general, return **all the crops** in the list, separated by a single space.  
6. Strictly follow this format: Return crop names with only a single space between them, no commas, no extra spaces at the beginning or end.  
7. Do not repeat any crop name in the list.  
8. If you're unsure of which crops are relevant, return all crops in the list, separated by a single space.  
9. Do not include anything other than the final space-separated answer.  
10. Do not provide any explanation, reasoning, or extra content.  
11. Do not provide any extra crop names if there are particular relevant crops to the query.

*FINAL ANSWER FORMAT EXAMPLE:*  

Final Answer Example:  
wheat jowar brinjal
"""
print(llm.invoke(prompt))
