import pandas as pd
import os

INPUT_PATH = "data/raw/healthcare.csv"
OUTPUT_PATH = "data/cleaned/cleaned_healthcare.csv"

os.makedirs("data/cleaned", exist_ok=True)

df = pd.read_csv(INPUT_PATH)
print("Before cleaning:", df.shape)

# Remove duplicates
df = df.drop_duplicates()
# Drop rows with any missing value
df = df.dropna()

# Standardise text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip().str.title()

# Convert dates (keep them for now, will be dropped in training)
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")

# Drop irrelevant columns (as required)
df = df.drop(columns=["Name", "Doctor", "Hospital", "Room Number"], errors="ignore")

print("After cleaning:", df.shape)
df.to_csv(OUTPUT_PATH, index=False)
print("✅ Cleaning complete")