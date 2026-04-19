import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

df = pd.read_csv("data/cleaned/cleaned_healthcare.csv")
df.to_sql("patients", engine, if_exists="replace", index=False)
print(f"Loaded {len(df)} rows into 'patients' table")