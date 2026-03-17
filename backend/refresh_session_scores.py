import sqlite3
from pathlib import Path

from api import SessionData, score_session


DB_PATH = Path(__file__).resolve().parent / "sessions.db"


def refresh_scores() -> dict[str, int]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, total_clicks, back_clicks, retry_count, dwell_time,
                   page_switch_count, user_id, entry_screen_id, exit_screen_id,
                   device_type, browser
            FROM sessions
            ORDER BY id
            """
        )
        rows = cur.fetchall()

        counts = {"Low": 0, "Medium": 0, "High": 0, "total": 0}
        for row in rows:
            session = SessionData(
                total_clicks=row["total_clicks"],
                back_clicks=row["back_clicks"],
                retry_count=row["retry_count"],
                dwell_time=row["dwell_time"],
                page_switch_count=row["page_switch_count"],
                user_id=row["user_id"],
                entry_screen_id=row["entry_screen_id"],
                exit_screen_id=row["exit_screen_id"],
                device_type=row["device_type"],
                browser=row["browser"],
            )
            prediction, probability, level = score_session(session)
            cur.execute(
                """
                UPDATE sessions
                SET friction_prediction = ?, friction_probability = ?, friction_level = ?
                WHERE id = ?
                """,
                (prediction, probability, level, row["id"]),
            )
            counts[level] += 1
            counts["total"] += 1

        conn.commit()
        return counts
    finally:
        conn.close()


if __name__ == "__main__":
    result = refresh_scores()
    print(
        f"Refreshed {result['total']} sessions. "
        f"Low={result['Low']} Medium={result['Medium']} High={result['High']}"
    )
