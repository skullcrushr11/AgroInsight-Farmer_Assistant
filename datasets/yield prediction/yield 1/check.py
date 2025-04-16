import pandas as pd

# Load the file
df = pd.read_csv('crop_yield.csv', sep='\t')  # Change sep=',' if comma-separated

# Print column names and first few rows
print("Column Names:", df.columns.tolist())
print("\nFirst 5 Rows:\n", df.head())