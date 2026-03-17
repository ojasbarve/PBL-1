import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from api import SessionData, get_connection, insert_session_record, score_session


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


@dataclass
class SessionAccumulator:
    session_id: str
    user_id: str | None = None
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    total_clicks: int = 0
    back_clicks: int = 0
    retry_count: int = 0
    page_switch_count: int = 0
    last_event_type: str | None = None
    last_product_id: str | None = None
    rows: list[tuple[datetime, str, str | None, str | None]] = field(default_factory=list)
    finalized: bool = False

    def update(self, row: dict[str, str]) -> None:
        timestamp = datetime.strptime(row["Timestamp"], TIMESTAMP_FORMAT)
        event_type = (row.get("EventType") or "").strip()
        product_id = (row.get("ProductID") or "").strip() or None
        user_id = (row.get("UserID") or "").strip() or None
        self.rows.append((timestamp, event_type, product_id, user_id))

    def finalize(self) -> None:
        if self.finalized:
            return

        self.rows.sort(key=lambda row: row[0])
        if self.rows:
            self.first_ts = self.rows[0][0]
            self.last_ts = self.rows[-1][0]
            self.user_id = self.rows[0][3]

        for _, event_type, product_id, _ in self.rows:
            self.total_clicks += 1

            if self.last_event_type is not None and event_type != self.last_event_type:
                self.page_switch_count += 1

            if self.last_event_type == event_type and self.last_product_id == product_id:
                self.retry_count += 1

            if event_type == "page_view" and self.last_event_type in {"product_view", "add_to_cart", "purchase"}:
                self.back_clicks += 1

            self.last_event_type = event_type
            self.last_product_id = product_id

        self.finalized = True

    def to_session_data(self) -> SessionData:
        self.finalize()
        dwell_time = 0
        if self.first_ts is not None and self.last_ts is not None:
            dwell_time = max(int((self.last_ts - self.first_ts).total_seconds()), 1)

        return SessionData(
            total_clicks=self.total_clicks,
            back_clicks=self.back_clicks,
            retry_count=self.retry_count,
            dwell_time=dwell_time,
            page_switch_count=max(self.page_switch_count, 1),
            user_id=f"kaggle-user-{self.user_id or self.session_id}",
            entry_screen_id=self.rows[0][1] if self.rows else None,
            exit_screen_id=self.rows[-1][1] if self.rows else None,
            device_type="imported_csv",
            browser="kaggle_clickstream",
        )


def load_sessions(csv_path: Path) -> dict[tuple[str, str], SessionAccumulator]:
    sessions: dict[str, SessionAccumulator] = {}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get("SessionID") or "").strip()
            if key not in sessions:
                sessions[key] = SessionAccumulator(session_id=key)
            sessions[key].update(row)

    return sessions


def import_sessions(csv_path: Path) -> dict[str, int]:
    sessions = load_sessions(csv_path)
    counts = defaultdict(int)

    for accumulator in sessions.values():
        session = accumulator.to_session_data()
        prediction, probability, level = score_session(session)
        insert_session_record(session, prediction, probability, level)
        counts[level] += 1
        counts["total"] += 1

    return counts


def clear_existing_imported_rows() -> int:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sessions WHERE browser = 'kaggle_clickstream'")
        existing = int(cursor.fetchone()[0])
        cursor.execute("DELETE FROM sessions WHERE browser = 'kaggle_clickstream'")
        conn.commit()
        return existing
    finally:
        conn.close()


if __name__ == "__main__":
    source = Path(r"C:\Users\Manan\Downloads\ecommerce_clickstream_10k.csv")
    removed = clear_existing_imported_rows()
    results = import_sessions(source)
    print(
        f"Removed {removed} previous imported rows. "
        f"Imported {results['total']} sessions from {source}. "
        f"Low={results['Low']} Medium={results['Medium']} High={results['High']}"
    )
