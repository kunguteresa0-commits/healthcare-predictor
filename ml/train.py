import os
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Read data
df = pd.read_sql("SELECT * FROM patients", engine)
df = df.dropna()

# Rename columns to match frontend (underscores)
df = df.rename(columns={
    "Blood Type": "Blood_Type",
    "Medical Condition": "Medical_Condition",
    "Billing Amount": "Billing_Amount",
    "Admission Type": "Admission_Type",
    "Insurance Provider": "Insurance_Provider",
    "Test Results": "Test_Results"
})

target_col = "Test_Results"
categorical_cols = ['Gender', 'Blood_Type', 'Medical_Condition', 'Insurance_Provider', 'Admission_Type', 'Medication']
numeric_cols = ['Age', 'Billing_Amount']

df = df[categorical_cols + numeric_cols + [target_col]]

# Encode categorical features
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"Encoded {col} -> {list(le.classes_)}")

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df[target_col])
print(f"Target classes: {list(target_encoder.classes_)}")

X = df[categorical_cols + numeric_cols]

# Scale numeric columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# XGBoost
xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

def evaluate(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    print(cm)
    return f1

f1_rf = evaluate(y_test, rf_pred, "Random Forest")
f1_xgb = evaluate(y_test, xgb_pred, "XGBoost")

os.makedirs("models", exist_ok=True)
if f1_rf >= f1_xgb:
    joblib.dump(rf, "models/model.joblib")
    print("Saved Random Forest")
else:
    joblib.dump(xgb, "models/model.joblib")
    print("Saved XGBoost")

joblib.dump(encoders, "models/encoders.joblib")
joblib.dump(target_encoder, "models/target_encoder.joblib")
joblib.dump(scaler, "models/scaler.joblib")
print("All artifacts saved.")

