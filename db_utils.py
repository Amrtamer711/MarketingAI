import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
import asyncio
import os
import json

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
                    "Reference Number" TEXT,
                    "Location" TEXT,
                    "Sales Person" TEXT,
                    "Submitted By" TEXT,
                    "Status" TEXT,
                    "Filming Date" TEXT,
                    "Videographer" TEXT,
                    "Task Type" TEXT DEFAULT 'videography',
                    "Submission Folder" TEXT,
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
                    task_type TEXT DEFAULT 'videography',
                    submission_folder TEXT,
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
            # Create approval workflows table for persistence across restarts
            conn.execute("""
                CREATE TABLE IF NOT EXISTS approval_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    task_number INTEGER,
                    folder_name TEXT,
                    dropbox_path TEXT,
                    videographer_id TEXT,
                    task_data TEXT,
                    version_info TEXT,
                    reviewer_id TEXT,
                    reviewer_msg_ts TEXT,
                    hos_id TEXT,
                    hos_msg_ts TEXT,
                    reviewer_approved INTEGER DEFAULT 0,
                    hos_approved INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT,
                    status TEXT DEFAULT 'pending'
                );
            """)
            # Helpful indexes for faster lookups
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_aw_task ON approval_workflows(task_number)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_aw_status ON approval_workflows(status)")
            except Exception as e:
                logger.warning(f"Index creation failed or already exists: {e}")
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


def tasks_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert task rows to DataFrame with proper column names."""
    if not rows:
        return pd.DataFrame(columns=[
            "Task #", "Timestamp", "Brand", "Campaign Start Date", "Campaign End Date",
            "Reference Number", "Location", "Sales Person", "Submitted By", "Status",
            "Filming Date", "Videographer", "Task Type", "Submission Folder", "Current Version",
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
                'Filming Date', 'Videographer', 'Task Type', 'Submission Folder', 'Current Version', 'Version History',
                'Pending Timestamps', 'Submitted Timestamps', 'Returned Timestamps', 'Rejected Timestamps', 'Accepted Timestamps'
            ]
            vals = [row.get(c, '') for c in cols]
            placeholders = ','.join(['?'] * len(cols))
            quoted_cols = [f'"{c}"' if ' ' in c else c for c in cols]
            conn.execute(f"INSERT INTO {LIVE_TABLE} ({','.join(quoted_cols)}) VALUES ({placeholders})", vals)
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
                        history = json.loads(vh) if isinstance(vh, str) else []
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
                    conn.execute(f"UPDATE {LIVE_TABLE} SET 'Version History'=? WHERE task_number=?", (json.dumps(history), task_number))
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
                    (task_number, brand, campaign_start_date, campaign_end_date, reference_number, location, sales_person, submitted_by, status, filming_date, videographer, task_type, submission_folder, current_version, version_history, pending_timestamps, submitted_timestamps, returned_timestamps, rejected_timestamps, accepted_timestamps, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row["task_number"], row["Brand"], row["Campaign Start Date"], row["Campaign End Date"], row["Reference Number"],
                    row["Location"], row["Sales Person"], row["Submitted By"], row["Status"], row["Filming Date"],
                    row["Videographer"], row["Task Type"] if "Task Type" in row.keys() else "videography", row["Submission Folder"] if "Submission Folder" in row.keys() else "", row["Current Version"], row["Version History"], row["Pending Timestamps"],
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
            live = conn.execute(f'SELECT * FROM {LIVE_TABLE} WHERE "Reference Number"=?', (clean_ref,)).fetchone()
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


# Async database operations
async def init_db_async():
    """Initialize the database asynchronously."""
    init_db()


async def save_task(data: Dict[str, Any]) -> Dict[str, Any]:
    """Save parsed campaign data to DB. Returns task_number."""
    try:
        from utils import calculate_filming_date
        
        # Ensure DB tables exist
        init_db()
        
        # Get next task number
        task_number = get_next_task_number()
        
        # Calculate filming date with new rules
        filming_date = calculate_filming_date(
            data.get("start_date", ""),
            data.get("end_date", "")
        )
        
        # Format dates to DD-MM-YYYY
        start_date = data.get("start_date", "")
        end_date = data.get("end_date", "")
        
        if start_date and len(start_date) == 10 and start_date[4] == '-':
            try:
                date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                start_date = date_obj.strftime("%d-%m-%Y")
            except:
                pass
        if end_date and len(end_date) == 10 and end_date[4] == '-':
            try:
                date_obj = datetime.strptime(end_date, "%Y-%m-%d")
                end_date = date_obj.strftime("%d-%m-%Y")
            except:
                pass
        
        row = {
            'task_number': task_number,
            'Timestamp': datetime.now(UAE_TZ).strftime("%d-%m-%Y %H:%M:%S"),
            'Brand': (data.get("brand", "") or '').replace("_", "-"),
            'Campaign Start Date': start_date,
            'Campaign End Date': end_date,
            'Reference Number': (data.get("reference_number", "") or '').replace("_", "-"),
            'Location': (data.get("location", "") or '').replace("_", "-"),
            'Sales Person': (data.get("sales_person", "") or '').replace("_", "-"),
            'Submitted By': (data.get("submitted_by", "") or '').replace("_", "-"),
            'Status': "Not assigned yet",
            'Filming Date': filming_date,
            'Videographer': "",
            'Task Type': data.get("task_type", "videography"),
            'Submission Folder': "",
            'Current Version': "",
            'Version History': "[]",
            'Pending Timestamps': "",
            'Submitted Timestamps': "",
            'Returned Timestamps': "",
            'Rejected Timestamps': "",
            'Accepted Timestamps': "",
        }
        insert_task(row)
        return {"success": True, "task_number": task_number}
    except Exception as e:
        logger.error(f"Error saving to DB: {e}")
        return {"success": False, "task_number": None}


async def get_all_tasks_df() -> pd.DataFrame:
    """Read all live tasks from DB and return as DataFrame."""
    try:
        init_db()
        rows = select_all_tasks()
        return tasks_to_dataframe(rows)
    except Exception as e:
        logger.error(f"DB read error: {e}")
        return tasks_to_dataframe([])


async def export_data_to_slack(include_history: bool = True, channel: str = None, user_id: str = None) -> str:
    """Export current tasks and optional history DB file to Slack."""
    from clients import slack_client
    from tempfile import NamedTemporaryFile
    
    if not channel or not user_id:
        return "❌ Channel and user information required to send files"
    
    response = "📊 **Exporting System Data Files...**\n\n"
    files_sent = []
    
    # Export live tasks as a generated Excel file
    try:
        df = await get_all_tasks_df()
        live_count = len(df)
        with NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name
        await asyncio.to_thread(df.to_excel, tmp_path, index=False)
        with open(tmp_path, 'rb') as f:
            filename = f"design_requests_{datetime.now(UAE_TZ).strftime('%Y%m%d_%H%M%S')}.xlsx"
            result = await slack_client.files_upload_v2(
                channel=channel,
                file=f,
                filename=filename,
                title=filename,
                initial_comment=f"📋 Excel file with {live_count} live tasks"
            )
        try:
            os.remove(tmp_path)
        except:
            pass
        if result.get('ok'):
            files_sent.append("Excel")
            response += f"✅ Excel file sent ({live_count} live tasks)\n"
        else:
            response += f"❌ Failed to send Excel file: {result.get('error', 'Unknown error')}\n"
    except Exception as e:
        logger.error(f"Error exporting live tasks: {e}")
        response += f"❌ Error exporting live tasks: {str(e)}\n"
    
    # Send complete database file if requested (includes both live and completed tasks)
    if include_history:
        try:
            if os.path.exists(HISTORY_DB_PATH):
                with _connect() as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM completed_tasks")
                    history_count = cursor.fetchone()[0]
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {LIVE_TABLE}")
                    live_in_db_count = cursor.fetchone()[0]
                with open(HISTORY_DB_PATH, 'rb') as f:
                    filename_db = f"complete_database_{datetime.now(UAE_TZ).strftime('%Y%m%d_%H%M%S')}.db"
                    result = await slack_client.files_upload_v2(
                        channel=channel,
                        file=f,
                        filename=filename_db,
                        title=filename_db,
                        initial_comment=f"🗄️ Complete database with {live_in_db_count} live tasks and {history_count} completed tasks"
                    )
                if result.get('ok'):
                    files_sent.append("Complete Database")
                    response += f"✅ Complete database sent ({live_in_db_count} live tasks, {history_count} completed tasks)\n"
                else:
                    response += f"❌ Failed to send complete database: {result.get('error', 'Unknown error')}\n"
            else:
                response += "⚠️ Database file not found\n"
        except Exception as e:
            logger.error(f"Error sending complete database: {e}")
            response += f"❌ Error sending complete database: {str(e)}\n"
    
    response += "\n" + "=" * 50 + "\n"
    if files_sent:
        response += f"\n✅ **Successfully exported {', '.join(files_sent)}**\n\n"
        response += "*How to view the files:*\n"
        response += "• *Excel (.xlsx)* - Open with Microsoft Excel, Google Sheets, or any spreadsheet app\n"
        response += "• *Database (.db)* - Open with SQLite browser or any SQLite viewer\n"
        response += "\n*Files contain sensitive business data - handle with care*"
    else:
        response += "\n❌ *No files were exported successfully*"
    response += f"\n\n_Export requested at {datetime.now(UAE_TZ).strftime('%d-%m-%Y %H:%M:%S')} UAE Time_"
    return response


async def get_task(task_number: int) -> Dict[str, Any]:
    """Get a specific task by task number from DB"""
    try:
        row = get_task_by_number(task_number)
        if not row:
            return None
        # Map to expected dict and format dates
        task_data = dict(row)
        # Normalize keys
        if 'task_number' in task_data and 'Task #' not in task_data:
            task_data['Task #'] = task_data.pop('task_number')
        for key, value in list(task_data.items()):
            if pd.isna(value):
                task_data[key] = ""
        # Format date-like fields
        for field in ['Campaign Start Date', 'Campaign End Date', 'Filming Date', 'Timestamp']:
            val = task_data.get(field)
            try:
                if val and isinstance(val, str):
                    # keep as-is if already dd-mm-yyyy
                    if '-' in val and len(val) >= 10:
                        continue
                if val:
                    dt = pd.to_datetime(val)
                    task_data[field] = dt.strftime('%d-%m-%Y')
            except:
                pass
        return task_data
    except Exception as e:
        logger.error(f"Error getting task {task_number}: {e}")
        return None


async def get_next_task_number_async() -> int:
    """Get the next available task number."""
    return get_next_task_number()


# ========== APPROVAL WORKFLOW PERSISTENCE ==========

def save_workflow(workflow_id: str, workflow_data: Dict[str, Any]) -> bool:
    """Save approval workflow to database"""
    try:
        with _connect() as conn:
            # Serialize complex data as JSON
            task_data_json = json.dumps(workflow_data.get('task_data', {}))
            version_info_json = json.dumps(workflow_data.get('version_info', {}))

            conn.execute("""
                INSERT OR REPLACE INTO approval_workflows
                (workflow_id, task_number, folder_name, dropbox_path, videographer_id,
                 task_data, version_info, reviewer_id, reviewer_msg_ts, hos_id,
                 hos_msg_ts, reviewer_approved, hos_approved, created_at, updated_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow_id,
                workflow_data.get('task_number'),
                workflow_data.get('folder_name'),
                workflow_data.get('dropbox_path'),
                workflow_data.get('videographer_id'),
                task_data_json,
                version_info_json,
                workflow_data.get('reviewer_id'),
                workflow_data.get('reviewer_msg_ts'),
                workflow_data.get('hos_id'),
                workflow_data.get('hos_msg_ts'),
                1 if workflow_data.get('reviewer_approved') else 0,
                1 if workflow_data.get('hos_approved') else 0,
                workflow_data.get('created_at', datetime.now(UAE_TZ).isoformat()),
                datetime.now(UAE_TZ).isoformat(),
                workflow_data.get('status', 'pending')
            ))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error saving workflow {workflow_id}: {e}")
        return False


def get_workflow(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Get approval workflow from database"""
    try:
        with _connect() as conn:
            cursor = conn.execute("""
                SELECT * FROM approval_workflows WHERE workflow_id = ?
            """, (workflow_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Convert row to dict
            columns = [desc[0] for desc in cursor.description]
            workflow = dict(zip(columns, row))
            
            # Deserialize JSON fields
            if workflow.get('task_data'):
                workflow['task_data'] = json.loads(workflow['task_data'])
            if workflow.get('version_info'):
                workflow['version_info'] = json.loads(workflow['version_info'])
            
            # Convert boolean fields
            workflow['reviewer_approved'] = bool(workflow.get('reviewer_approved'))
            workflow['hos_approved'] = bool(workflow.get('hos_approved'))
            
            return workflow
    except Exception as e:
        logger.error(f"Error getting workflow {workflow_id}: {e}")
        return None


def delete_workflow(workflow_id: str) -> bool:
    """Delete approval workflow from database"""
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM approval_workflows WHERE workflow_id = ?", (workflow_id,))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error deleting workflow {workflow_id}: {e}")
        return False


def get_all_pending_workflows() -> List[Dict[str, Any]]:
    """Get all pending workflows (for recovery on startup)"""
    try:
        with _connect() as conn:
            cursor = conn.execute("""
                SELECT * FROM approval_workflows 
                WHERE status = 'pending'
                ORDER BY created_at ASC
            """)
            rows = cursor.fetchall()
            
            workflows = []
            columns = [desc[0] for desc in cursor.description]
            
            for row in rows:
                workflow = dict(zip(columns, row))
                
                # Deserialize JSON fields
                if workflow.get('task_data'):
                    workflow['task_data'] = json.loads(workflow['task_data'])
                if workflow.get('version_info'):
                    workflow['version_info'] = json.loads(workflow['version_info'])
                
                # Convert boolean fields
                workflow['reviewer_approved'] = bool(workflow.get('reviewer_approved'))
                workflow['hos_approved'] = bool(workflow.get('hos_approved'))
                
                workflows.append(workflow)
            
            return workflows
    except Exception as e:
        logger.error(f"Error getting pending workflows: {e}")
        return []


async def save_workflow_async(workflow_id: str, workflow_data: Dict[str, Any]) -> bool:
    """Async wrapper for save_workflow"""
    return await asyncio.get_event_loop().run_in_executor(None, save_workflow, workflow_id, workflow_data)


async def get_workflow_async(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Async wrapper for get_workflow"""
    return await asyncio.get_event_loop().run_in_executor(None, get_workflow, workflow_id)


async def delete_workflow_async(workflow_id: str) -> bool:
    """Async wrapper for delete_workflow"""
    return await asyncio.get_event_loop().run_in_executor(None, delete_workflow, workflow_id)


async def get_all_pending_workflows_async() -> List[Dict[str, Any]]:
    """Async wrapper for get_all_pending_workflows"""
    return await asyncio.get_event_loop().run_in_executor(None, get_all_pending_workflows)


async def check_duplicate_async(reference_number: str) -> Dict[str, Any]:
    """Check if reference number already exists."""
    return check_duplicate_reference(reference_number)


async def delete_task_by_number(task_number: int) -> Dict[str, Any]:
    """Delete a task by task number (archive into history then remove from live)"""
    try:
        # Get task data first to return it and check if it has a Trello card
        task_data = get_task_by_number(task_number)
        if not task_data:
            return {"success": False, "error": "Task not found"}

        # First update the task status to "Archived"
        with _connect() as conn:
            conn.execute(f"UPDATE {LIVE_TABLE} SET Status='Archived' WHERE task_number=?", (task_number,))

        # Archive the Trello card if task was assigned to a videographer
        trello_archived = False
        if str(task_data.get('Status', '')).startswith('Assigned to'):
            try:
                from trello_utils import get_trello_card_by_task_number, archive_trello_card
                logger.info(f"Looking for Trello card for Task #{task_number}")

                card = await asyncio.to_thread(get_trello_card_by_task_number, task_number)

                if card:
                    logger.info(f"Found Trello card '{card['name']}' (ID: {card['id']})")
                    success = await asyncio.to_thread(archive_trello_card, card['id'])
                    if success:
                        logger.info(f"✅ Archived Trello card for Task #{task_number}")
                        trello_archived = True
                    else:
                        logger.warning(f"⚠️ Failed to archive Trello card for Task #{task_number}")
                else:
                    logger.info(f"No Trello card found for Task #{task_number}")
            except Exception as e:
                logger.error(f"Error archiving Trello card for Task #{task_number}: {e}")

        # Then archive the task in database
        ok = archive_task(task_number)
        if ok:
            # Return task data with trello status
            return {
                "success": True,
                "task_data": task_data,
                "trello_archived": trello_archived
            }
        return {"success": False, "error": "Failed to archive task"}
    except Exception as e:
        logger.error(f"Error deleting task {task_number}: {e}")
        return {"success": False, "error": str(e)}


async def update_task(task_number: int, updates: Dict[str, Any], current_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Update a task by task number in DB, including Trello if already assigned"""
    try:
        from trello_utils import get_trello_card_by_task_number, update_trello_card, get_trello_lists, update_checklist_dates
        
        # Get current data
        if not current_data:
            current_row = get_task_by_number(task_number)
            if not current_row:
                return {"success": False, "error": "Task not found"}
            current_data = dict(current_row)
            current_data['Task #'] = current_data.get('task_number', task_number)
        
        # Check if task has a Trello card (assigned or in video workflow)
        status = str(current_data.get('Status', ''))
        has_trello_card = (
            status.startswith('Assigned to') or 
            status in ['Critique', 'Editing', 'Submitted to Sales', 'Returned', 'Done']
        )
        trello_updates_needed = False
        trello_updates = {}
        
        # If has Trello card, prepare Trello updates
        if has_trello_card:
            if 'Videographer' in updates and updates['Videographer'] != current_data.get('Videographer'):
                # Only update status if it's currently "Assigned to X"
                if status.startswith('Assigned to'):
                    updates['Status'] = f"Assigned to {updates['Videographer']}"
                trello_updates_needed = True
                trello_updates['assignee'] = updates['Videographer']
            if 'Status' in updates and updates['Status'] != current_data.get('Status'):
                trello_updates_needed = True
                # Only extract assignee if the new status is "Assigned to X"
                if updates['Status'].startswith('Assigned to'):
                    new_assignee = updates['Status'].replace('Assigned to ', '')
                    trello_updates['assignee'] = new_assignee
            if 'Filming Date' in updates and updates['Filming Date'] != current_data.get('Filming Date'):
                trello_updates_needed = True
                try:
                    filming_date = pd.to_datetime(updates['Filming Date'])
                    trello_updates['filming_date'] = filming_date
                except:
                    pass
            detail_fields = ['Brand', 'Campaign Start Date', 'Campaign End Date', 'Reference Number', 'Location', 'Sales Person']
            details_changed = any(field in updates and updates[field] != current_data.get(field) for field in detail_fields)
            if details_changed or 'Videographer' in updates:
                trello_updates_needed = True
        
        # Check if status is being changed to Permanently Rejected
        if updates.get('Status') == 'Permanently Rejected':
            # Import the permanent rejection handler
            from video_upload_system import handle_permanent_rejection
            
            # Update status first so it's recorded properly
            updates['Rejected Timestamps'] = datetime.now(UAE_TZ).strftime('%d-%m-%Y %H:%M:%S')
            ok = update_task_by_number(task_number, updates)
            if not ok:
                return {"success": False, "error": "DB update failed"}
            
            # Handle permanent rejection (moves videos, cancels workflows, notifies)
            await handle_permanent_rejection(task_number, current_data)
            
            # Archive the task instead of just updating
            ok = archive_task(task_number)
            if not ok:
                return {"success": False, "error": "Failed to archive permanently rejected task"}
            
            return {"success": True, "updates": updates, "archived": True}
        
        # Persist DB updates for normal cases
        ok = update_task_by_number(task_number, updates)
        if not ok:
            return {"success": False, "error": "DB update failed"}
        
        # Trello updates
        if has_trello_card and trello_updates_needed:
            trello_card = get_trello_card_by_task_number(task_number)
            if trello_card:
                updated_data = current_data.copy()
                updated_data.update(updates)
                description = f"""Task #{task_number}
Brand: {updated_data.get('Brand', '')}
Campaign Start Date: {updated_data.get('Campaign Start Date', '')}
Campaign End Date: {updated_data.get('Campaign End Date', '')}
Reference: {updated_data.get('Reference Number', '')}
Location: {updated_data.get('Location', '')}
Sales Person: {updated_data.get('Sales Person', '')}
Videographer: {updated_data.get('Videographer', '')}"""
                trello_payload = {'desc': description}
                if 'Brand' in updates or 'Reference Number' in updates:
                    trello_payload['name'] = f"Task #{task_number}: {updated_data.get('Brand', '')} - {updated_data.get('Reference Number', '')}"
                if 'assignee' in trello_updates:
                    lists = get_trello_lists()
                    new_list_id = lists.get(trello_updates['assignee'])
                    if new_list_id:
                        trello_payload['idList'] = new_list_id
                if 'filming_date' in trello_updates:
                    update_checklist_dates(trello_card['id'], trello_updates['filming_date'])
                if trello_payload:
                    success = update_trello_card(trello_card['id'], trello_payload)
                    if not success:
                        return {"success": True, "warning": "DB updated but Trello update failed"}
            else:
                return {"success": True, "warning": "DB updated but Trello card not found"}
        
        return {"success": True, "updates": updates}
    except Exception as e:
        logger.error(f"Error updating task {task_number}: {e}")
        return {"success": False, "error": str(e)} 