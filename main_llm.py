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
login(token = "HuggingFace_API_KEY")

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

user_prompt=input("Enter your rice Query:")

results = vector_store_map['rice'].similarity_search(
    user_prompt,
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
prompt = f"""You are an expert agricultural assistant with deep knowledge in paddy (rice) farming...  

*CONTEXT:*  
{context}  

*QUESTION:*  
What is the optimal temperature for growing rice? 

*Instructions:*  

1. Carefully review the provided context.  
2. Break down the question into relevant parts if needed.  
3. Base your answer solely on the context and reliable paddy farming practices.  
4. Provide a concise, accurate, and detailed answer that addresses the question fully.  
5. Strictly print every new point in a new line.  
6. Strictly print only the final answer.  
7. Even if you are unsure, the user should not know that.  
8. Don't print the chain of thought on the screen.  
9. Do not print the same sentence twice.
10. Do not miss any points.  
11. Always Start with a heading with a small introduction.
12. The context given may include relevant json snippets, and should be handled accordingly

*FINAL ANSWER FORMAT EXAMPLE:*  

Final Answer Example:
Heading: Use appropriate introduction to the answer

- Use high-quality seeds to ensure better yield.  
- Maintain a proper water level of 5-7 cm during the vegetative stage.  
- Apply nitrogen fertilizers in three split doses for optimal growth.  
- Use pest control measures such as neem oil or pheromone traps.  


 *FINAL ANSWER:*
 """
print(llm.invoke(prompt))