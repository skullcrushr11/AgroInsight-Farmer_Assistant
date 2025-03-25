import os
from langchain.document_loaders import PyMuPDFLoader  # Fast & lightweight PDF parser

# Define input and output directories
pdf_folder = "PDF data/paddy"
output_folder = "Extracted Text"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Process each PDF file in the folder
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):  # Ensure only PDFs are processed
        pdf_path = os.path.join(pdf_folder, pdf_file)
        
        # Load and extract text
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        
        # Combine text from all pages
        extracted_text = "\n\n".join(doc.page_content for doc in docs)
        
        # Define output text file path
        text_file_name = os.path.splitext(pdf_file)[0] + ".txt"
        text_file_path = os.path.join(output_folder, text_file_name)
        
        # Save extracted text as a Notepad (.txt) file
        with open(text_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(extracted_text)
        
        print(f"Extracted text saved: {text_file_path}")

print("✅ All PDFs processed successfully!")

