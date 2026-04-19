from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Literal
import os
import traceback

class PatientInput(BaseModel):
    Age: int
    Gender: Literal["Male", "Female"]
    Blood_Type: str
    Medical_Condition: str
    Billing_Amount: float
    Admission_Type: Literal["Emergency", "Elective", "Urgent"]
    Insurance_Provider: str
    Medication: str

model = None
encoders = None
target_encoder = None
scaler = None

# IMPORTANT: This order must match the training order (categoricals first, then numerics)
feature_columns = [
    'Gender', 'Blood_Type', 'Medical_Condition', 'Insurance_Provider',
    'Admission_Type', 'Medication', 'Age', 'Billing_Amount'
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoders, target_encoder, scaler
    model = joblib.load("models/model.joblib")
    encoders = joblib.load("models/encoders.joblib")
    target_encoder = joblib.load("models/target_encoder.joblib")
    scaler = joblib.load("models/scaler.joblib")
    print("✅ Model and encoders loaded")
    print("Feature order:", feature_columns)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def frontend():
    with open("Frontend/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(patient: PatientInput):
    try:
        data = patient.dict()
        # Build a single row in the exact order
        row = []
        for col in feature_columns:
            if col in ['Age', 'Billing_Amount']:
                row.append(data[col])
            else:
                # categorical column: encode using saved encoder
                le = encoders[col]
                row.append(le.transform([data[col]])[0])
        
        X_input = np.array([row])
        # Scale the numeric columns (last two positions)
        X_input[:, -2:] = scaler.transform(X_input[:, -2:])
        
        pred_num = model.predict(X_input)[0]
        pred_label = target_encoder.inverse_transform([pred_num])[0]
        proba = model.predict_proba(X_input)[0].tolist()
        classes = target_encoder.classes_.tolist()
        
        return {
            "predicted_test_result": pred_label,
            "probabilities": dict(zip(classes, proba))
        }
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))