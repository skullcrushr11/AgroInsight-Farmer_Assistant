import pandas as pd

# Load the Excel file
file_path = "Crop_recommendation.csv"  # Change this to your file path
df = pd.read_csv(file_path)

# Specify the column name
column_name = "label"  # Change this to your actual column name

# Get unique values
unique_values = df[column_name].dropna().unique()

# Print the unique values
print("Unique values in column '{}':".format(column_name))
for value in unique_values:
    print(value)
