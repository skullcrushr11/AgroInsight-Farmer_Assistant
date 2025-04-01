import pandas as pd

# Load the CSV file
df = pd.read_csv("Crop and fertilizer dataset.csv")

# Specify the column name
column_name = "Soil_color"

# Print unique values
print(df[column_name].unique())
