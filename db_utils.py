import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd

from config import HISTORY_DB_PATH, UAE_TZ
from logger import logger

LIVE_TABLE = 'live_tasks'


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(HISTORY_DB_PATH)
    try:
        # Improve concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
    except Exception as e:
        logger.warning(f"PRAGMA setup failed: {e}")
    return conn


def init_db() -> None:
    try:
        with _connect() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {LIVE_TABLE} (
                    task_number INTEGER PRIMARY KEY AUTOINCREMENT,
                    "Timestamp" TEXT,
                    "Brand" TEXT,
                    "Campaign Start Date" TEXT,
                    "Campaign End Date" TEXT,
                    "Reference Number" TEXT UNIQUE,
                    "Location" TEXT,
                    "Sales Person" TEXT,
                    "Submitted By" TEXT,
                    "Status" TEXT,
                    "Filming Date" TEXT,
                    "Videographer" TEXT,
                    "Video Filename" TEXT,
                    "Current Version" TEXT,
                    "Version History" TEXT,
                    "Pending Timestamps" TEXT,
                    "Submitted Timestamps" TEXT,
                    "Returned Timestamps" TEXT,
                    "Rejected Timestamps" TEXT,
                    "Accepted Timestamps" TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS completed_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_number INTEGER,
                    brand TEXT,
                    campaign_start_date TEXT,
                    campaign_end_date TEXT,
                    reference_number TEXT,
                    location TEXT,
                    sales_person TEXT,
                    submitted_by TEXT,
                    status TEXT,
                    filming_date TEXT,
                    videographer TEXT,
                    current_version TEXT,
                    version_history TEXT,
                    pending_timestamps TEXT,
                    submitted_timestamps TEXT,
                    returned_timestamps TEXT,
                    rejected_timestamps TEXT,
                    accepted_timestamps TEXT,
                    completed_at TEXT
                );
            """)
            # Seed sequence so live task_number keeps increasing beyond history
            try:
                cur = conn.execute(f"SELECT COALESCE(MAX(task_number), 0) FROM {LIVE_TABLE}")
                max_live = cur.fetchone()[0] or 0
                cur = conn.execute("SELECT COALESCE(MAX(task_number), 0) FROM completed_tasks")
                max_hist = cur.fetchone()[0] or 0
                target = max(max_live, max_hist)
                if target > 0:
                    # Ensure sqlite_sequence updated
                    try:
                        existing = conn.execute("SELECT seq FROM sqlite_sequence WHERE name=?", (LIVE_TABLE,)).fetchone()
                        if existing is None:
                            conn.execute("INSERT INTO sqlite_sequence(name, seq) VALUES (?, ?)", (LIVE_TABLE, target))
                        elif (existing[0] or 0) < target:
                            conn.execute("UPDATE sqlite_sequence SET seq=? WHERE name=?", (target, LIVE_TABLE))
                    except Exception:
                        # Fallback: insert and delete a dummy row at target to bump sequence
                        try:
                            conn.execute(f"INSERT INTO {LIVE_TABLE}(task_number) VALUES (?)", (target,))
                            conn.execute(f"DELETE FROM {LIVE_TABLE} WHERE task_number=?", (target,))
                        except Exception as e2:
                            logger.warning(f"Sequence seed fallback failed: {e2}")
            except Exception as e:
                logger.warning(f"Sequence seed check failed: {e}")
    except Exception as e:
        logger.error(f"DB init error: {e}")


def rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[
            "Task #", "Timestamp", "Brand", "Campaign Start Date", "Campaign End Date",
            "Reference Number", "Location", "Sales Person", "Submitted By", "Status",
            "Filming Date", "Videographer", "Video Filename", "Current Version",
            "Version History", "Pending Timestamps", "Submitted Timestamps",
            "Returned Timestamps", "Rejected Timestamps", "Accepted Timestamps"
        ])
    df = pd.DataFrame(rows)
    df.rename(columns={"task_number": "Task #"}, inplace=True)
    return df


def select_all_tasks() -> List[Dict[str, Any]]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"SELECT * FROM {LIVE_TABLE}").fetchall()
        return [dict(r) for r in rows]


def get_next_task_number() -> int:
    try:
        with _connect() as conn:
            live_max = conn.execute(f"SELECT MAX(task_number) FROM {LIVE_TABLE}").fetchone()[0] or 0
            hist_max = conn.execute("SELECT MAX(task_number) FROM completed_tasks").fetchone()[0] or 0
            return max(live_max, hist_max) + 1
    except Exception as e:
        logger.error(f"DB get_next_task_number error: {e}")
        return 1


def insert_task(row: Dict[str, Any]) -> int:
    init_db()
    try:
        with _connect() as conn:
            cols = [
                'Timestamp', 'Brand', 'Campaign Start Date', 'Campaign End Date',
                'Reference Number', 'Location', 'Sales Person', 'Submitted By', 'Status',
                'Filming Date', 'Videographer', 'Video Filename', 'Current Version', 'Version History',
                'Pending Timestamps', 'Submitted Timestamps', 'Returned Timestamps', 'Rejected Timestamps', 'Accepted Timestamps'
            ]
            vals = [row.get(c, '') for c in cols]
            placeholders = ','.join(['?'] * len(cols))
            conn.execute(f"INSERT INTO {LIVE_TABLE} ({','.join([f'"{c}"' if ' ' in c else c for c in cols])}) VALUES ({placeholders})", vals)
            task_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            return int(task_id)
    except Exception as e:
        logger.error(f"DB insert_task error: {e}")
        raise


def get_task_by_number(task_number: int) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(f"SELECT * FROM {LIVE_TABLE} WHERE task_number=?", (task_number,)).fetchone()
        return dict(row) if row else None


def update_task_by_number(task_number: int, updates: Dict[str, Any]) -> bool:
    try:
        with _connect() as conn:
            sets = []
            vals = []
            for k, v in updates.items():
                col = f'"{k}"' if ' ' in k and k != 'task_number' else k
                if k == 'Task #':
                    continue
                sets.append(f"{col}=?")
                vals.append(v)
            if not sets:
                return True
            vals.append(task_number)
            conn.execute(f"UPDATE {LIVE_TABLE} SET {', '.join(sets)} WHERE task_number=?", vals)
            return True
    except Exception as e:
        logger.error(f"DB update_task_by_number error: {e}")
        return False


def update_status_with_history_and_timestamp(task_number: int, folder: str, version: Optional[int] = None,
                                              rejection_reason: Optional[str] = None, rejection_class: Optional[str] = None,
                                              rejected_by: Optional[str] = None) -> bool:
    folder_to_status = {
        "Raw": "Raw", "Pending": "Critique", "Rejected": "Editing",
        "Submitted to Sales": "Submitted to Sales", "Accepted": "Done", "Returned": "Returned",
        "raw": "Raw", "pending": "Critique", "rejected": "Editing", "submitted": "Submitted to Sales",
        "accepted": "Done", "returned": "Returned"
    }
    new_status = folder_to_status.get(folder, "Unknown")
    folder_to_column = {
        "pending": "Pending Timestamps", "Pending": "Pending Timestamps", "Critique": "Pending Timestamps",
        "submitted": "Submitted Timestamps", "Submitted to Sales": "Submitted Timestamps",
        "returned": "Returned Timestamps", "Returned": "Returned Timestamps",
        "rejected": "Rejected Timestamps", "Rejected": "Rejected Timestamps", "Editing": "Rejected Timestamps",
        "accepted": "Accepted Timestamps", "Accepted": "Accepted Timestamps", "Done": "Accepted Timestamps"
    }
    try:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("BEGIN IMMEDIATE;")
                row = conn.execute(f"SELECT * FROM {LIVE_TABLE} WHERE task_number=?", (task_number,)).fetchone()
                if not row:
                    conn.execute("ROLLBACK;")
                    return False
                # Update status
                conn.execute(f"UPDATE {LIVE_TABLE} SET 'Status'=? WHERE task_number=?", (new_status, task_number))
                # Version history
                if version is not None:
                    vh = row["Version History"] or '[]'
                    try:
                        history = pd.io.json.loads(vh) if isinstance(vh, str) else []
                    except Exception:
                        history = []
                    event_time = datetime.now(UAE_TZ).strftime("%d-%m-%Y %H:%M:%S")
                    entry = {"version": version, "folder": folder, "at": event_time}
                    if (folder.lower() in ["rejected", "returned"]) and rejection_class:
                        entry["rejection_class"] = rejection_class
                        entry["rejection_comments"] = rejection_reason or ""
                        if rejected_by:
                            entry["rejected_by"] = rejected_by
                    history.append(entry)
                    conn.execute(f"UPDATE {LIVE_TABLE} SET 'Version History'=? WHERE task_number=?", (pd.io.json.dumps(history), task_number))
                # Movement timestamp
                col = folder_to_column.get(folder) or folder_to_column.get(new_status)
                if col:
                    existing = row[col] or ''
                    ts = datetime.now(UAE_TZ).strftime("%d-%m-%Y %H:%M:%S")
                    stamp = f"v{version}:{ts}" if version is not None else ts
                    updated = (existing + ("; " if existing else "") + stamp)
                    conn.execute(f"UPDATE {LIVE_TABLE} SET '{col}'=? WHERE task_number=?", (updated, task_number))
                conn.execute("COMMIT;")
                return True
            except Exception as e:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
                logger.error(f"DB update_status_with_history error: {e}")
                return False
    except Exception as e:
        logger.error(f"DB update_status_with_history error: {e}")
        return False


def archive_task(task_number: int) -> bool:
    try:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("BEGIN IMMEDIATE;")
                row = conn.execute(f"SELECT * FROM {LIVE_TABLE} WHERE task_number=?", (task_number,)).fetchone()
                if not row:
                    conn.execute("ROLLBACK;")
                    return False
                conn.execute("""
                    INSERT INTO completed_tasks
                    (task_number, brand, campaign_start_date, campaign_end_date, reference_number, location, sales_person, submitted_by, status, filming_date, videographer, current_version, version_history, pending_timestamps, submitted_timestamps, returned_timestamps, rejected_timestamps, accepted_timestamps, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["task_number"], row["Brand"], row["Campaign Start Date"], row["Campaign End Date"], row["Reference Number"],
                    row["Location"], row["Sales Person"], row["Submitted By"], row["Status"], row["Filming Date"],
                    row["Videographer"], row["Current Version"], row["Version History"], row["Pending Timestamps"],
                    row["Submitted Timestamps"], row["Returned Timestamps"], row["Rejected Timestamps"], row["Accepted Timestamps"],
                    datetime.now(UAE_TZ).strftime('%d-%m-%Y %H:%M:%S')
                ))
                conn.execute(f"DELETE FROM {LIVE_TABLE} WHERE task_number=?", (task_number,))
                conn.execute("COMMIT;")
                return True
            except Exception as e:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
                logger.error(f"DB archive_task error: {e}")
                return False
    except Exception as e:
        logger.error(f"DB archive_task error: {e}")
        return False


def check_duplicate_reference(reference_number: str) -> Dict[str, Any]:
    clean_ref = reference_number.replace('_', '-')
    try:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            live = conn.execute(f"SELECT * FROM {LIVE_TABLE} WHERE 'Reference Number'=?", (clean_ref,)).fetchone()
            if live:
                return {"is_duplicate": True, "existing_entry": {
                    "task_number": str(live["task_number"]),
                    "brand": live["Brand"],
                    "start_date": live["Campaign Start Date"],
                    "end_date": live["Campaign End Date"],
                    "location": live["Location"],
                    "submitted_by": live["Submitted By"],
                    "timestamp": live["Timestamp"],
                    "status": "Active"
                }}
            hist = conn.execute("""
                SELECT task_number, brand, campaign_start_date, campaign_end_date, location, submitted_by, completed_at
                FROM completed_tasks WHERE reference_number=?
            """, (clean_ref,)).fetchone()
            if hist:
                return {"is_duplicate": True, "existing_entry": {
                    "task_number": str(hist[0]),
                    "brand": hist[1] or '',
                    "start_date": hist[2] or '',
                    "end_date": hist[3] or '',
                    "location": hist[4] or '',
                    "submitted_by": hist[5] or '',
                    "timestamp": hist[6] or '',
                    "date": hist[2] or '',
                    "status": "Archived (Completed)"
                }}
            return {"is_duplicate": False}
    except Exception as e:
        logger.error(f"DB check_duplicate_reference error: {e}")
        return {"is_duplicate": False} 