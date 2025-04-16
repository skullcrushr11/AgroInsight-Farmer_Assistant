import pandas as pd

# Load the CSV file
df = pd.read_csv("fertilizer_recommendation_dataset.csv")

# Specify the column name
column_name = "Soil"

# Print unique values
print(df[column_name].unique())
