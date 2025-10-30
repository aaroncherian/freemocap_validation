import sqlite3


DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY,
    participant_code TEXT NOT NULL, 
    trial_name TEXT NOT NULL,
    trial_type TEXT NOT NULL,
    trial_number TEXT NOT NULL,
    session_date TEXT,
    data_root TEXT,
    notes TEXT,
    UNIQUE (participant_code, trial_name)
    );

    CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY,
    trial_id INTEGER NOT NULL REFERENCES trials(id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    condition TEXT NOT NULL DEFAULT '',
    tracker TEXT NOT NULL,
    component_name TEXT NOT NULL,
    path TEXT NOT NULL,
    file_exists INTEGER NOT NULL,
    size_bytes INTEGER,
    mtime_utc REAL,
    UNIQUE (trial_id, category, condition, tracker, component_name)
)
"""

def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(DDL)
    return conn

def upsert_trial(conn: sqlite3.Connection,
                 participant_code: str,
                 trial_name: str,
                 trial_type: str,
                 trial_number: str,
                 session_date: str | None = None,
                    data_root: str | None = None,
                    notes: str | None = None) -> int:
    sql = """
    INSERT INTO trials (participant_code, trial_name, trial_type, trial_number, session_date, data_root, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(participant_code, trial_name) DO UPDATE SET
        trial_type     = excluded.trial_type,
        trial_number   = excluded.trial_number,
        session_date   = excluded.session_date,
        data_root      = excluded.data_root,
        notes          = excluded.notes;
    """

    conn.execute(sql, (participant_code, trial_name, trial_type, trial_number, session_date, data_root, notes))
    cur = conn.execute("SELECT id FROM trials WHERE participant_code = ? AND trial_name = ?", (participant_code, trial_name))
    return cur.fetchone()[0] #gets the trial_id so we can use it when inserting artifacts

def upsert_artifact(conn: sqlite3.Connection,
                    trial_id: int,
                    rows):
    sql = """
    INSERT INTO artifacts (trial_id, category, condition, tracker, component_name, path, file_exists, size_bytes, mtime_utc)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(trial_id, category, condition, tracker, component_name) DO UPDATE SET
        path = excluded.path,
        file_exists = excluded.file_exists,
        size_bytes = excluded.size_bytes,
        mtime_utc = excluded.mtime_utc
    """

    data = [
        (trial_id,
        r["category"],
        r["condition"],
        r["tracker"],
        r["component_name"],
        r["path"],
        r["file_exists"],
        r["size_bytes"],
        r["mtime_utc"])
        for r in rows
    ]

    conn.executemany(sql, data)
