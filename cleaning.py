import re

# Define the allowed special characters
allowed_special_chars = ",.><='%" 

def clean_text(text, lowercase=True):
    """
    Cleans the input text by removing special characters except allowed ones.
    Converts text to lowercase if specified.
    """
    # Remove all characters except letters, numbers, spaces, and allowed special chars
    text = re.sub(r"[^a-zA-Z0-9\s,.><='%:]", "", text)
    
    # Convert to lowercase if required
    if lowercase:
        text = text.lower()
    
    return text

# Process the file
input_file = "combined.txt"  # Replace with your actual file
output_file = "cleaned_text.txt"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        cleaned_line = clean_text(line, lowercase=True)  # Change to False if using cased BERT
        outfile.write(cleaned_line + "\n")

print("Cleaning complete! Output saved to", output_file)
