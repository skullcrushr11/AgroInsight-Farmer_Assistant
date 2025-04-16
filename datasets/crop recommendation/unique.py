import pandas as pd

# Load the CSV file
df = pd.read_csv("Crop_Recommendation.csv")

# Specify the column name
column_name = "label"

# Print unique values
print(df[column_name].unique())
