from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel
import sqlite3

# Load trained model
model = joblib.load("friction_model.pkl")

app = FastAPI()
def init_db():
    conn = sqlite3.connect("sessions.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        total_clicks INTEGER,
        back_clicks INTEGER,
        retry_count INTEGER,
        dwell_time INTEGER,
        page_switch_count INTEGER,
        friction_prediction INTEGER,
        friction_probability REAL,
        friction_level TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

# Initialize database on startup
init_db()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input structure
class SessionData(BaseModel):
    total_clicks: int
    back_clicks: int
    retry_count: int
    dwell_time: int
    page_switch_count: int

@app.get("/")
def home():
    return {"message": "Friction Detection API is running"}

@app.post("/predict")
def predict_friction(data: SessionData):
    
    input_data = np.array([[
        data.total_clicks,
        data.back_clicks,
        data.retry_count,
        data.dwell_time,
        data.page_switch_count
    ]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Determine friction level
    if probability < 0.33:
        level = "Low"
    elif probability < 0.66:
        level = "Medium"
    else:
        level = "High"

    # 🔥 INSERT INTO DATABASE
    conn = sqlite3.connect("sessions.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO sessions (
            total_clicks,
            back_clicks,
            retry_count,
            dwell_time,
            page_switch_count,
            friction_prediction,
            friction_probability,
            friction_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.total_clicks,
        data.back_clicks,
        data.retry_count,
        data.dwell_time,
        data.page_switch_count,
        int(prediction),
        float(probability),
        level
    ))

    conn.commit()
    conn.close()

    return {
        "friction_prediction": int(prediction),
        "friction_probability": float(probability),
        "friction_level": level
    }


@app.get("/sessions")
def get_sessions():
    conn = sqlite3.connect("sessions.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, total_clicks, back_clicks, retry_count,
               dwell_time, page_switch_count,
               friction_probability, friction_level, timestamp
        FROM sessions
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    sessions = []
    for row in rows:
        sessions.append({
            "id": row[0],
            "total_clicks": row[1],
            "back_clicks": row[2],
            "retry_count": row[3],
            "dwell_time": row[4],
            "page_switch_count": row[5],
            "friction_probability": row[6],
            "friction_level": row[7],
            "timestamp": row[8]
        })

    return sessions

@app.get("/analytics-summary")
def analytics_summary():
    conn = sqlite3.connect("sessions.db")
    cursor = conn.cursor()

    # Total sessions
    cursor.execute("SELECT COUNT(*) FROM sessions")
    total_sessions = cursor.fetchone()[0]

    # Count friction levels
    cursor.execute("""
        SELECT friction_level, COUNT(*)
        FROM sessions
        GROUP BY friction_level
    """)
    level_counts = cursor.fetchall()

    level_dict = {"Low": 0, "Medium": 0, "High": 0}

    for level, count in level_counts:
        level_dict[level] = count

    # Average friction probability
    cursor.execute("SELECT AVG(friction_probability) FROM sessions")
    avg_probability = cursor.fetchone()[0]

    conn.close()

    return {
        "total_sessions": total_sessions,
        "low_friction": level_dict["Low"],
        "medium_friction": level_dict["Medium"],
        "high_friction": level_dict["High"],
        "average_friction_score": round(avg_probability if avg_probability else 0, 3)
    }