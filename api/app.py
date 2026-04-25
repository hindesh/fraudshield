import os
import uuid
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from db import init_db, insert_transaction, get_all_transactions, get_stats

load_dotenv()

app = Flask(__name__)
CORS(app)

print("🧠 Loading ML model...")
model        = joblib.load("models/fraud_model.pkl")
encoders     = joblib.load("models/encoders.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")
print("✅ Model loaded successfully")

init_db()

def get_risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

def get_reason_codes(data: dict, probability: float) -> list:
    reasons = []

    if data.get("TransactionAmt", 0) > 500:
        reasons.append("high_transaction_amount")

    if data.get("P_emaildomain", "") in ["anonymous.com", "protonmail.com"]:
        reasons.append("suspicious_email_domain")

    if data.get("card6", "") == "debit":
        reasons.append("debit_card_used")

    if data.get("addr1", 0) == 0 or data.get("addr2", 0) == 0:
        reasons.append("missing_address")

    if probability >= 0.7:
        reasons.append("high_model_confidence")

    if not reasons:
        reasons.append("model_pattern_detection")

    return reasons

def preprocess_input(data: dict) -> pd.DataFrame:
    # Build a row with all expected feature columns, defaulting to 0
    row = {col: 0 for col in feature_cols}

    # Map incoming fields to feature columns
    field_map = {
        "TransactionAmt" : "TransactionAmt",
        "ProductCD"      : "ProductCD",
        "card1"          : "card1",
        "card2"          : "card2",
        "card4"          : "card4",
        "card6"          : "card6",
        "addr1"          : "addr1",
        "addr2"          : "addr2",
        "P_emaildomain"  : "P_emaildomain",
        "dist1"          : "dist1",
        "C1"             : "C1",
        "C2"             : "C2",
        "C6"             : "C6",
        "C13"            : "C13",
        "V258"           : "V258",
        "V257"           : "V257",
        "V201"           : "V201",
    }

    for input_key, feature_key in field_map.items():
        if input_key in data and feature_key in row:
            row[feature_key] = data[input_key]

    df = pd.DataFrame([row])

    # Encode categorical columns using saved encoders
    cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain"]
    for col in cat_cols:
        if col in df.columns and col in encoders:
            le = encoders[col]
            val = str(df[col].iloc[0])
            # Handle unseen labels gracefully
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                df[col] = 0

    return df


# Health Check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"  : "ok",
        "service" : "fraudshield",
        "model"   : "xgboost_v1"
    }), 200


# Score a Transaction
@app.route("/score", methods=["POST"])
def score():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Preprocess
        df = preprocess_input(data)

        # Predict
        fraud_probability = float(model.predict_proba(df)[:, 1][0])
        risk_level        = get_risk_level(fraud_probability)
        is_flagged        = fraud_probability >= 0.5
        reason_codes      = get_reason_codes(data, fraud_probability)
        transaction_id    = f"txn_{uuid.uuid4().hex[:12]}"

        # Log to DB
        insert_transaction({
            "transaction_id"    : transaction_id,
            "transaction_amt"   : data.get("TransactionAmt"),
            "product_cd"        : data.get("ProductCD"),
            "card4"             : data.get("card4"),
            "p_emaildomain"     : data.get("P_emaildomain"),
            "device_type"       : data.get("DeviceType"),
            "fraud_probability" : fraud_probability,
            "risk_level"        : risk_level,
            "is_flagged"        : is_flagged,
            "reason_codes"      : reason_codes
        })

        return jsonify({
            "transaction_id"    : transaction_id,
            "fraud_probability" : round(fraud_probability, 4),
            "risk_level"        : risk_level,
            "is_flagged"        : is_flagged,
            "reason_codes"      : reason_codes
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Get All Transactions
@app.route("/transactions", methods=["GET"])
def transactions():
    try:
        limit = request.args.get("limit", 50, type=int)
        rows  = get_all_transactions(limit)
        return jsonify({
            "count"        : len(rows),
            "transactions" : rows
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Get Stats
@app.route("/stats", methods=["GET"])
def stats():
    try:
        data = get_stats()
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)