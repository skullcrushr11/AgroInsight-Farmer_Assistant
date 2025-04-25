import re


allowed_special_chars = ",.><='%" 

def clean_text(text, lowercase=True):
    """
    Cleans the input text by removing special characters except allowed ones.
    Converts text to lowercase if specified.
    """
    
    text = re.sub(r"[^a-zA-Z0-9\s,.><='%:]", "", text)
    
    
    if lowercase:
        text = text.lower()
    
    return text


input_file = "combined.txt"  
output_file = "cleaned_text.txt"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        cleaned_line = clean_text(line, lowercase=True)  
        outfile.write(cleaned_line + "\n")

print("Cleaning complete! Output saved to", output_file)
