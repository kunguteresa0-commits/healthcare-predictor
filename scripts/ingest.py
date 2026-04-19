import os
import shutil

SOURCE = "healthcare_dataset.csv"   # your downloaded file
DEST_DIR = "data/raw"
DEST_PATH = os.path.join(DEST_DIR, "healthcare.csv")

os.makedirs(DEST_DIR, exist_ok=True)
shutil.copy(SOURCE, DEST_PATH)
print("✅ Dataset copied to data/raw/healthcare.csv")