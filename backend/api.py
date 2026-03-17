from contextlib import closing
import hashlib
from html import escape
from pathlib import Path
import sqlite3

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "sessions.db"
MODEL_PATH = BASE_DIR / "friction_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)

app = FastAPI(title="Friction Detection API")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(cursor: sqlite3.Cursor, column_name: str, definition: str) -> None:
    cursor.execute("PRAGMA table_info(sessions)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if column_name not in existing_columns:
        cursor.execute(f"ALTER TABLE sessions ADD COLUMN {column_name} {definition}")


def init_db() -> None:
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_clicks INTEGER NOT NULL,
                back_clicks INTEGER NOT NULL,
                retry_count INTEGER NOT NULL,
                dwell_time INTEGER NOT NULL,
                page_switch_count INTEGER NOT NULL,
                friction_prediction INTEGER NOT NULL,
                friction_probability REAL NOT NULL,
                friction_level TEXT NOT NULL,
                user_id TEXT,
                entry_screen_id TEXT,
                exit_screen_id TEXT,
                device_type TEXT,
                browser TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        ensure_column(cursor, "user_id", "TEXT")
        ensure_column(cursor, "entry_screen_id", "TEXT")
        ensure_column(cursor, "exit_screen_id", "TEXT")
        ensure_column(cursor, "device_type", "TEXT")
        ensure_column(cursor, "browser", "TEXT")
        ensure_column(cursor, "timestamp", "DATETIME DEFAULT CURRENT_TIMESTAMP")
        conn.commit()


init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionData(BaseModel):
    total_clicks: int
    back_clicks: int
    retry_count: int
    dwell_time: int
    page_switch_count: int
    user_id: str | None = None
    entry_screen_id: str | None = None
    exit_screen_id: str | None = None
    device_type: str | None = None
    browser: str | None = None


def build_model_features(data: SessionData) -> np.ndarray:
    retry_ratio = data.retry_count / (data.total_clicks + 1)
    back_click_ratio = data.back_clicks / (data.total_clicks + 1)
    avg_dwell_time = data.dwell_time / (data.page_switch_count + 1)
    raw_features = np.array([[retry_ratio, back_click_ratio, avg_dwell_time]])
    return SCALER.transform(raw_features)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def event_adjustment(event_name: str | None) -> float:
    if not event_name:
        return 0.0
    adjustments = {
        "page_view": -0.03,
        "product_view": 0.02,
        "click": 0.03,
        "add_to_cart": 0.08,
        "login": 0.04,
        "logout": -0.02,
        "purchase": 0.12,
        "checkout": 0.10,
        "payment": 0.10,
        "search": 0.03,
        "details": 0.02,
    }
    return adjustments.get(event_name.lower(), 0.0)


def metadata_jitter(data: SessionData) -> float:
    seed = "|".join(
        str(part or "")
        for part in (
            data.user_id,
            data.entry_screen_id,
            data.exit_screen_id,
            data.device_type,
            data.browser,
            data.total_clicks,
            data.back_clicks,
            data.retry_count,
            data.dwell_time,
            data.page_switch_count,
        )
    )
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return ((digest[0] / 255.0) - 0.5) * 0.08


def heuristic_probability(data: SessionData) -> float:
    total_clicks = max(data.total_clicks, 1)
    click_pressure = clamp(data.total_clicks / 18.0)
    back_pressure = clamp(data.back_clicks / max(total_clicks * 0.35, 1.0))
    retry_pressure = clamp(data.retry_count / max(total_clicks * 0.25, 1.0))
    dwell_pressure = clamp(data.dwell_time / 480.0)
    switch_pressure = clamp(data.page_switch_count / 10.0)

    score = (
        0.05
        + click_pressure * 0.18
        + back_pressure * 0.22
        + retry_pressure * 0.24
        + dwell_pressure * 0.16
        + switch_pressure * 0.08
        + event_adjustment(data.entry_screen_id) * 0.5
        + event_adjustment(data.exit_screen_id) * 0.7
        + metadata_jitter(data)
    )
    return clamp(score, 0.05, 0.95)


def score_session(data: SessionData) -> tuple[int, float, str]:
    input_data = build_model_features(data)
    model_probability = float(MODEL.predict_proba(input_data)[0][1])
    probability = clamp((model_probability * 0.35) + (heuristic_probability(data) * 0.65), 0.05, 0.95)
    prediction = 1 if probability >= 0.68 else 0

    if probability < 0.34:
        level = "Low"
    elif probability < 0.68:
        level = "Medium"
    else:
        level = "High"

    return prediction, probability, level


def insert_session_record(
    data: SessionData,
    prediction: int,
    probability: float,
    level: str,
) -> int:
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (
                total_clicks,
                back_clicks,
                retry_count,
                dwell_time,
                page_switch_count,
                friction_prediction,
                friction_probability,
                friction_level,
                user_id,
                entry_screen_id,
                exit_screen_id,
                device_type,
                browser
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.total_clicks,
                data.back_clicks,
                data.retry_count,
                data.dwell_time,
                data.page_switch_count,
                prediction,
                probability,
                level,
                data.user_id,
                data.entry_screen_id,
                data.exit_screen_id,
                data.device_type,
                data.browser,
            ),
        )
        session_id = cursor.lastrowid
        conn.commit()
    return session_id


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "Friction Detection API is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "database": str(DB_PATH)}


@app.post("/predict")
def predict_friction(data: SessionData) -> dict[str, float | int | str | None]:
    prediction, probability, level = score_session(data)
    session_id = insert_session_record(data, prediction, probability, level)

    return {
        "id": session_id,
        "friction_prediction": prediction,
        "friction_probability": probability,
        "friction_level": level,
        "user_id": data.user_id,
        "entry_screen_id": data.entry_screen_id,
        "exit_screen_id": data.exit_screen_id,
        "device_type": data.device_type,
        "browser": data.browser,
    }


@app.get("/sessions")
def get_sessions() -> list[dict[str, str | float | int | None]]:
    with closing(get_connection()) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, total_clicks, back_clicks, retry_count,
                   dwell_time, page_switch_count,
                   friction_prediction, friction_probability, friction_level,
                   user_id, entry_screen_id, exit_screen_id, device_type, browser,
                   timestamp
            FROM sessions
            ORDER BY id DESC
            """
        )
        rows = cursor.fetchall()

    return [dict(row) for row in rows]


@app.get("/analytics-summary")
def analytics_summary() -> dict[str, float | int]:
    with closing(get_connection()) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) AS total_sessions FROM sessions")
        total_sessions = int(cursor.fetchone()[0])

        cursor.execute(
            """
            SELECT friction_level, COUNT(*) AS count
            FROM sessions
            GROUP BY friction_level
            """
        )
        level_counts = cursor.fetchall()

        level_dict = {"Low": 0, "Medium": 0, "High": 0}
        for level, count in level_counts:
            level_dict[level] = count

        cursor.execute("SELECT AVG(friction_probability) AS avg_probability FROM sessions")
        avg_probability = cursor.fetchone()[0]

    return {
        "total_sessions": total_sessions,
        "low_friction": level_dict["Low"],
        "medium_friction": level_dict["Medium"],
        "high_friction": level_dict["High"],
        "average_friction_score": round(avg_probability if avg_probability else 0, 3),
    }


def render_html_table(rows: list[dict[str, str | float | int | None]]) -> str:
    if not rows:
        return "<p>No rows found.</p>"

    columns = list(rows[0].keys())
    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body_rows = []

    for row in rows:
        cells = "".join(
            f"<td>{escape('' if row[column] is None else str(row[column]))}</td>"
            for column in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")

    return (
        "<div class='table-wrap'>"
        "<table>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def admin_page_template(title: str, body: str) -> HTMLResponse:
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>{escape(title)}</title>
        <style>
            body {{
                margin: 0;
                font-family: Arial, sans-serif;
                background: #f4f7fb;
                color: #142033;
            }}
            .shell {{
                max-width: 1280px;
                margin: 0 auto;
                padding: 24px;
            }}
            .card {{
                background: #fff;
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 12px 30px rgba(20, 32, 51, 0.08);
                margin-bottom: 20px;
            }}
            h1, h2, p {{
                margin-top: 0;
            }}
            .meta {{
                color: #5f6f86;
            }}
            .links a {{
                margin-right: 16px;
                color: #1f6feb;
                text-decoration: none;
                font-weight: 600;
            }}
            .table-wrap {{
                overflow: auto;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                min-width: 960px;
            }}
            th, td {{
                border: 1px solid #d7dfeb;
                padding: 10px 12px;
                text-align: left;
                white-space: nowrap;
            }}
            th {{
                background: #1c2a40;
                color: #f4f7fb;
                position: sticky;
                top: 0;
            }}
            tr:nth-child(even) td {{
                background: #f9fbfe;
            }}
            code {{
                background: #eef3fa;
                padding: 2px 6px;
                border-radius: 6px;
            }}
        </style>
    </head>
    <body>
        <div class="shell">
            {body}
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/admin", response_class=HTMLResponse)
def admin_home() -> HTMLResponse:
    summary = analytics_summary()
    body = f"""
    <div class="card">
        <h1>Backend Admin</h1>
        <p class="meta">Database engine: <code>SQLite</code></p>
        <p class="meta">Database file: <code>{escape(str(DB_PATH))}</code></p>
        <div class="links">
            <a href="/admin/sessions">View Sessions Table</a>
            <a href="/admin/summary">View Summary Table</a>
            <a href="/sessions">Raw JSON Sessions</a>
        </div>
    </div>
    <div class="card">
        <h2>Current Summary</h2>
        <p>Total sessions: <strong>{summary["total_sessions"]}</strong></p>
        <p>Low: <strong>{summary["low_friction"]}</strong> | Medium: <strong>{summary["medium_friction"]}</strong> | High: <strong>{summary["high_friction"]}</strong></p>
        <p>Average friction score: <strong>{summary["average_friction_score"]}</strong></p>
    </div>
    """
    return admin_page_template("Backend Admin", body)


@app.get("/admin/sessions", response_class=HTMLResponse)
def admin_sessions() -> HTMLResponse:
    rows = get_sessions()
    body = f"""
    <div class="card">
        <h1>Stored Sessions</h1>
        <p class="meta">Rows from <code>sessions</code> in <code>{escape(str(DB_PATH))}</code></p>
        <div class="links">
            <a href="/admin">Admin Home</a>
            <a href="/sessions">Raw JSON</a>
        </div>
    </div>
    <div class="card">
        {render_html_table(rows)}
    </div>
    """
    return admin_page_template("Stored Sessions", body)


@app.get("/admin/summary", response_class=HTMLResponse)
def admin_summary() -> HTMLResponse:
    summary = analytics_summary()
    rows = [summary]
    body = f"""
    <div class="card">
        <h1>Analytics Summary</h1>
        <p class="meta">Aggregated values computed from the stored <code>sessions</code> table.</p>
        <div class="links">
            <a href="/admin">Admin Home</a>
            <a href="/analytics-summary">Raw JSON</a>
        </div>
    </div>
    <div class="card">
        {render_html_table(rows)}
    </div>
    """
    return admin_page_template("Analytics Summary", body)
