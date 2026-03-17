import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "sessions.db"


def rebalance() -> dict[str, int]:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, friction_probability
            FROM sessions
            ORDER BY friction_probability ASC, id ASC
            """
        )
        rows = cur.fetchall()
        total = len(rows)
        if total == 0:
            return {"total": 0, "low": 0, "medium": 0, "high": 0}

        low_cutoff = total // 3
        medium_cutoff = (2 * total) // 3

        low_ids = [row[0] for row in rows[:low_cutoff]]
        medium_ids = [row[0] for row in rows[low_cutoff:medium_cutoff]]
        high_ids = [row[0] for row in rows[medium_cutoff:]]

        def update_ids(ids: list[int], level: str, prediction: int) -> None:
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

        update_ids(low_ids, "Low", 0)
        update_ids(medium_ids, "Medium", 0)
        update_ids(high_ids, "High", 1)
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
    result = rebalance()
    print(
        f"Rebalanced {result['total']} sessions. "
        f"Low={result['low']} Medium={result['medium']} High={result['high']}"
    )
