import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

print("📦 Loading dataset...")
df = pd.read_csv("train_transaction.csv")

FEATURES = [
    "TransactionAmt",
    "ProductCD",
    "card1", "card2", "card4", "card6",
    "addr1", "addr2",
    "P_emaildomain",
    "dist1",
    "C1", "C2", "C6", "C13",
    "V258", "V257", "V201"
]
TARGET = "isFraud"

df = df[FEATURES + [TARGET]]

print("🧹 Cleaning data...")

# Fill numeric columns with median
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != TARGET]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical columns with "unknown"
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
df[cat_cols] = df[cat_cols].fillna("unknown")

print("🔠 Encoding categorical features...")
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  # save for use during inference

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"📊 Dataset shape: {X.shape}")
print(f"🔍 Fraud ratio: {y.mean()*100:.2f}%")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training XGBoost model...")

# scale_pos_weight handles class imbalance automatically
fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=fraud_ratio,   # handles imbalance
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

print("\n Evaluating model...")
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nSaving model...")
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")
joblib.dump(list(X.columns), "models/feature_columns.pkl")

print("✅ Model saved to models/fraud_model.pkl")
print("✅ Encoders saved to models/encoders.pkl")
print("✅ Feature columns saved to models/feature_columns.pkl")