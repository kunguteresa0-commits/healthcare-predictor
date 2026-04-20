## 🏥 Healthcare Test Result Predictor

A simple end-to-end machine learning healthcare system that predicts whether a patient's test result will be:

- ✅ Normal
- ⚠️ Abnormal
- ❓ Inconclusive

The system cleans healthcare data, trains a model, and provides predictions through a FastAPI web API and user interface.

## 🎯 Project Goal

Healthcare systems generate large amounts of patient data, but analyzing it manually is slow.

This project solves that by:

- Cleaning raw healthcare data
- Storing it properly
- Training a machine learning model
- Allowing users to predict patient test results instantly
## 🧠 How the System Works
1. Data Ingestion
Dataset downloaded from Kaggle
Stored as raw data (healthcare_dataset.csv)
2. Data Cleaning

Handled in scripts/clean.py:

- Removed duplicates
- Handled missing values
- Standardized text (e.g. Male/Female)
- Converted data types
- Dropped unnecessary columns
3. Machine Learning Training

Handled in ml/train.py:

Encoded categorical data
Scaled numeric values
Trained models:
- XGBoost
- Random Forest
Evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
Saved model files:
- model.joblib
- encoders.joblib
- scaler.joblib
- target_encoder.joblib
4. Model Storage

All trained files are saved in:

models/

These are used later by the API for predictions.

5. API Development (FastAPI)

Handled in:

- app/main.py
Available Endpoints:
Endpoint	Method	Description
/	GET	Web interface
/predict	POST	Predict patient test result
Example Request:
{
  "Age": 45,
  "Gender": "Male",
  "Blood Type": "O+",
  "Medical Condition": "Diabetes",
  "Billing Amount": 2000.5,
  "Admission Type": "Emergency",
  "Insurance Provider": "Cigna",
  "Medication": "Aspirin"
}
Example Response:
{
  "predicted_test_result": "Abnormal"
}
6. Frontend

A simple interface built using HTML:

Frontend/index.html
Users input patient details
Click Predict
See results instantly
7. Deployment

The API is deployed online using Render:

🔗 Your Live API:
👉 https://healthcare-predictor-3skk.onrender.com

🛠️ Tech Stack
Layer	Tool
API	FastAPI
ML	Scikit-learn, XGBoost
Data	Pandas, NumPy
Model Saving	Joblib
Frontend	HTML
Deployment	Render
Package Manager	UV
## 📂 Project Structure
## PRE-INTERNSHIP-HEALTHCARE-MACHINE-LEARNING/

├── app/
│   └── main.py

├── data/
│   ├── raw/
│   └── cleaned/

├── ml/
│   └── train.py

├── models/
│   ├── model.joblib
│   ├── encoders.joblib
│   ├── scaler.joblib
│   └── target_encoder.joblib

├── scripts/
│   ├── ingest.py
│   ├── clean.py

├── Frontend/
│   └── index.html

├── healthcare_dataset.csv
├── pyproject.toml
├── render.yaml
├── docker-compose.yml
├── README.md
## ⚙️ How to Run the Project Locally
1. Install UV
pip install uv
2. Create Virtual Environment
uv venv

Activate:

.venv\Scripts\activate
3. Install Dependencies
uv pip install -r requirements.txt
4. Run Data Pipeline
python scripts/ingest.py
python scripts/clean.py
5. Train Model
python ml/train.py
6. Start API
uvicorn app.main:app --reload

Open in browser:

http://localhost:8000
🧪 Testing the API

You can test using:

Swagger UI:
http://localhost:8000/docs
Or Postman

Try different inputs like:

Young vs old patients
Different medical conditions
Different billing amounts
## 🚀 Key Features
Full data pipeline (raw → cleaned → model)
Machine learning with multiple models
Real-time predictions
Web interface for easy use
Deployed live API
Uses UV instead of pip
## 📊 Notes
Dataset is synthetic (from Kaggle)
Accuracy may not be very high due to random data
Focus is on system design and pipeline, not medical accuracy
## 🎓 Skills Demonstrated
Data Cleaning & Preprocessing
Machine Learning (XGBoost, Random Forest)
API Development (FastAPI)
Model Serialization
Deployment (Render)
Git & GitHub
Debugging real-world errors
## 🙌 Final Note

This project demonstrates how to build a complete machine learning system from scratch to deployment, using real tools used in industry.
