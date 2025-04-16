import pandas as pd

# Load the CSV file
df = pd.read_csv("crop_yield.csv")

# Specify the column name
column_name = "Crop"

# Print unique values
print(df[column_name].unique())
