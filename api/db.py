import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        database=os.getenv("DB_NAME", "fraudshield"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres")
    )

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id                  SERIAL PRIMARY KEY,
            transaction_id      VARCHAR(50) UNIQUE NOT NULL,
            transaction_amt     FLOAT,
            product_cd          VARCHAR(10),
            card4               VARCHAR(20),
            p_emaildomain       VARCHAR(100),
            device_type         VARCHAR(20),
            fraud_probability   FLOAT,
            risk_level          VARCHAR(10),
            is_flagged          BOOLEAN,
            reason_codes        TEXT[],
            created_at          TIMESTAMP DEFAULT NOW()
        );
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Database initialized successfully")

def insert_transaction(data: dict):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO transactions (
            transaction_id,
            transaction_amt,
            product_cd,
            card4,
            p_emaildomain,
            device_type,
            fraud_probability,
            risk_level,
            is_flagged,
            reason_codes
        ) VALUES (
            %(transaction_id)s,
            %(transaction_amt)s,
            %(product_cd)s,
            %(card4)s,
            %(p_emaildomain)s,
            %(device_type)s,
            %(fraud_probability)s,
            %(risk_level)s,
            %(is_flagged)s,
            %(reason_codes)s
        )
        ON CONFLICT (transaction_id) DO NOTHING;
    """, data)

    conn.commit()
    cursor.close()
    conn.close()

def get_all_transactions(limit: int = 50):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT * FROM transactions
        ORDER BY created_at DESC
        LIMIT %s;
    """, (limit,))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [dict(row) for row in rows]

def get_stats():
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("""
        SELECT
            COUNT(*)                                        AS total_transactions,
            SUM(CASE WHEN is_flagged THEN 1 ELSE 0 END)    AS total_flagged,
            AVG(fraud_probability)                          AS avg_fraud_probability,
            SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END)   AS high_risk,
            SUM(CASE WHEN risk_level = 'MEDIUM' THEN 1 ELSE 0 END) AS medium_risk,
            SUM(CASE WHEN risk_level = 'LOW' THEN 1 ELSE 0 END)    AS low_risk
        FROM transactions;
    """)

    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return dict(row)