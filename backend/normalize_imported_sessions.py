import sqlite3
from pathlib import Path

from api import SessionData, score_session


DB_PATH = Path(__file__).resolve().parent / "sessions.db"


def realistic_dwell_time(total_clicks: int, back_clicks: int, retry_count: int, page_switch_count: int) -> int:
    estimated = (
        25
        + total_clicks * 18
        + back_clicks * 14
        + retry_count * 22
        + page_switch_count * 12
    )
    return max(45, min(estimated, 900))


def normalize_imported_sessions() -> dict[str, int]:
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
            WHERE browser = 'kaggle_clickstream'
            ORDER BY id
            """
        )
        rows = cur.fetchall()
        if not rows:
            return {"total": 0, "low": 0, "medium": 0, "high": 0}

        rescored: list[tuple[int, int, float]] = []
        for row in rows:
            dwell_time = realistic_dwell_time(
                row["total_clicks"],
                row["back_clicks"],
                row["retry_count"],
                row["page_switch_count"],
            )
            session = SessionData(
                total_clicks=row["total_clicks"],
                back_clicks=row["back_clicks"],
                retry_count=row["retry_count"],
                dwell_time=dwell_time,
                page_switch_count=row["page_switch_count"],
                user_id=row["user_id"],
                entry_screen_id=row["entry_screen_id"],
                exit_screen_id=row["exit_screen_id"],
                device_type=row["device_type"],
                browser=row["browser"],
            )
            prediction, probability, _ = score_session(session)
            rescored.append((row["id"], dwell_time, float(probability)))

            cur.execute(
                """
                UPDATE sessions
                SET dwell_time = ?, friction_prediction = ?, friction_probability = ?
                WHERE id = ?
                """,
                (dwell_time, prediction, float(probability), row["id"]),
            )

        rescored.sort(key=lambda item: (item[2], item[0]))
        total = len(rescored)
        low_cutoff = int(total * 0.4)
        medium_cutoff = int(total * 0.75)

        low_ids = [row_id for row_id, _, _ in rescored[:low_cutoff]]
        medium_ids = [row_id for row_id, _, _ in rescored[low_cutoff:medium_cutoff]]
        high_ids = [row_id for row_id, _, _ in rescored[medium_cutoff:]]

        def update_level(ids: list[int], level: str, prediction: int) -> None:
            if not ids:
                return
            placeholders = ",".join("?" for _ in ids)
            cur.execute(
                f"""
                UPDATE sessions
                SET friction_level = ?, friction_prediction = ?
                WHERE id IN ({placeholders})
                """,
                [level, prediction, *ids],
            )

        update_level(low_ids, "Low", 0)
        update_level(medium_ids, "Medium", 0)
        update_level(high_ids, "High", 1)
        conn.commit()

        return {
            "total": total,
            "low": len(low_ids),
            "medium": len(medium_ids),
            "high": len(high_ids),
        }
    finally:
        conn.close()


if __name__ == "__main__":
    result = normalize_imported_sessions()
    print(
        f"Normalized {result['total']} imported sessions. "
        f"Low={result['low']} Medium={result['medium']} High={result['high']}"
    )
