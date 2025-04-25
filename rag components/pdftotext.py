import os
from langchain.document_loaders import PyMuPDFLoader  


pdf_folder = "PDF data/paddy"
output_folder = "Extracted Text"


os.makedirs(output_folder, exist_ok=True)


for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):  
        pdf_path = os.path.join(pdf_folder, pdf_file)
        
        
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        
        
        extracted_text = "\n\n".join(doc.page_content for doc in docs)
        
        
        text_file_name = os.path.splitext(pdf_file)[0] + ".txt"
        text_file_path = os.path.join(output_folder, text_file_name)
        
        
        with open(text_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(extracted_text)
        
        print(f"Extracted text saved: {text_file_path}")

print("✅ All PDFs processed successfully!")

