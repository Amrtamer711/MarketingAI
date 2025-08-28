import asyncio
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font

from config import EXCEL_FILE_PATH, UAE_TZ, HISTORY_DB_PATH
from logger import logger
from trello_utils import get_trello_card_by_task_number, update_trello_card, get_trello_lists
from utils import calculate_filming_date

# New DB utilities for live tasks
from db_utils import (
    init_db as db_init,
    rows_to_df as db_rows_to_df,
    select_all_tasks as db_select_all_tasks,
    get_next_task_number as db_next_task_number,
    insert_task as db_insert_task,
    get_task_by_number as db_get_task_by_number,
    update_task_by_number as db_update_task_by_number,
    check_duplicate_reference as db_check_duplicate_reference,
)

# ========== ASYNC EXCEL MANAGEMENT ==========
async def initialize_excel():
    """Create Excel file with headers if it doesn't exist (legacy)"""
    # DB is now the source of truth; ensure DB initialized
    try:
        db_init()
    except Exception as e:
        logger.error(f"DB init failed: {e}")

async def save_to_excel(data: Dict[str, Any]) -> Dict[str, Any]:
    """Save parsed campaign data to DB (replacing Excel). Returns task_number."""
    try:
        # Ensure DB tables exist
        db_init()
        
        # Get next task number
        task_number = db_next_task_number()
        
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
            'Video Filename': "",
            'Current Version': "",
            'Version History': "[]",
            'Pending Timestamps': "",
            'Submitted Timestamps': "",
            'Returned Timestamps': "",
            'Rejected Timestamps': "",
            'Accepted Timestamps': "",
        }
        db_insert_task(row)
        return {"success": True, "task_number": task_number}
    except Exception as e:
        logger.error(f"Error saving to DB: {e}")
        return {"success": False, "task_number": None}

async def read_excel_async() -> pd.DataFrame:
    """Read live tasks from DB and return as DataFrame (Excel-compatible columns)."""
    try:
        db_init()
        rows = db_select_all_tasks()
        return db_rows_to_df(rows)
    except Exception as e:
        logger.error(f"DB read error: {e}")
        return db_rows_to_df([])

async def export_current_data(include_history: bool = True, format: str = "files", channel: str = None, user_id: str = None) -> str:
    """Export current live tasks (from DB) and optional history DB file to Slack."""
    try:
        import sqlite3
        from config import HISTORY_DB_PATH
        from clients import slack_client
        from tempfile import NamedTemporaryFile
        
        if not channel or not user_id:
            return "❌ Channel and user information required to send files"
        
        response = "📊 **Exporting System Data Files...**\n\n"
        files_sent = []
        
        # Export live tasks as a generated Excel file
        try:
            df = await read_excel_async()
            live_count = len(df)
            with NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                tmp_path = tmp.name
            await asyncio.to_thread(df.to_excel, tmp_path, False)
            with open(tmp_path, 'rb') as f:
                result = await slack_client.files_upload_v2(
                    channel=channel,
                    file=f,
                    filename=f"design_requests_{datetime.now(UAE_TZ).strftime('%Y%m%d_%H%M%S')}.xlsx",
                    title="Current Live Tasks (from DB)",
                    initial_comment=f"📋 **Excel file with {live_count} live tasks**"
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
        
        # Send History DB file if requested
        if include_history:
            try:
                if os.path.exists(HISTORY_DB_PATH):
                    with sqlite3.connect(HISTORY_DB_PATH) as conn:
                        cursor = conn.execute("SELECT COUNT(*) FROM completed_tasks")
                        history_count = cursor.fetchone()[0]
                    with open(HISTORY_DB_PATH, 'rb') as f:
                        result = await slack_client.files_upload_v2(
                            channel=channel,
                            file=f,
                            filename=f"history_logs_{datetime.now(UAE_TZ).strftime('%Y%m%d_%H%M%S')}.db",
                            title="History Database (Completed Tasks)",
                            initial_comment=f"✅ **History database with {history_count} completed tasks**"
                        )
                    if result.get('ok'):
                        files_sent.append("History DB")
                        response += f"✅ History database sent ({history_count} completed tasks)\n"
                    else:
                        response += f"❌ Failed to send history database: {result.get('error', 'Unknown error')}\n"
                else:
                    response += "⚠️ History database not found\n"
            except Exception as e:
                logger.error(f"Error sending history database: {e}")
                response += f"❌ Error sending history database: {str(e)}\n"
        
        response += "\n" + "=" * 50 + "\n"
        if files_sent:
            response += f"\n✅ **Successfully exported {', '.join(files_sent)}**\n\n"
            response += "**How to view the files:**\n"
            response += "• **Excel (.xlsx)** - Open with Microsoft Excel, Google Sheets, or any spreadsheet app\n"
            response += "• **Database (.db)** - Open with SQLite browser or any SQLite viewer\n"
            response += "\n_Files contain sensitive business data - handle with care_"
        else:
            response += "\n❌ **No files were exported successfully**"
        response += f"\n\n_Export requested at {datetime.now(UAE_TZ).strftime('%d-%m-%Y %H:%M:%S')} UAE Time_"
        return response
    except Exception as e:
        logger.error(f"Error exporting data files: {e}")
        return f"❌ Error exporting data files: {str(e)}"

async def get_task_by_number(task_number: int) -> Dict[str, Any]:
    """Get a specific task by task number from DB"""
    try:
        row = db_get_task_by_number(task_number)
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

async def get_next_task_number() -> int:
    """Delegate to DB for next task number"""
    try:
        return db_next_task_number()
    except Exception as e:
        logger.warning(f"Error reading history for task number: {e}")
        return 1

async def check_duplicate_reference(reference_number: str) -> Dict[str, Any]:
    """Delegate duplicate reference check to DB across live and history"""
    return db_check_duplicate_reference(reference_number)

async def update_movement_timestamp(task_number: int, folder: str, version: Optional[int] = None):
    """Deprecated: movement timestamps are now updated via DB status update path"""
    logger.info("update_movement_timestamp is deprecated in DB mode")

async def delete_task_by_number(task_number: int) -> Dict[str, Any]:
    """Delete a task by task number (archive into history then remove from live)"""
    try:
        from db_utils import archive_task
        ok = archive_task(task_number)
        if ok:
            return {"success": True, "task_data": {"Task #": task_number}}
        return {"success": False, "error": "Task not found"}
    except Exception as e:
        logger.error(f"Error deleting task {task_number}: {e}")
        return {"success": False, "error": str(e)}

async def update_task_by_number(task_number: int, updates: Dict[str, Any], current_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Update a task by task number in DB, including Trello if already assigned"""
    try:
        # Get current data
        if not current_data:
            current_row = db_get_task_by_number(task_number)
            if not current_row:
                return {"success": False, "error": "Task not found"}
            current_data = dict(current_row)
            current_data['Task #'] = current_data.get('task_number', task_number)
        
        # Check assignment
        is_assigned = str(current_data.get('Status', '')).startswith('Assigned to')
        trello_updates_needed = False
        trello_updates = {}
        
        # If assigned, prepare Trello updates
        if is_assigned:
            if 'Videographer' in updates and updates['Videographer'] != current_data.get('Videographer'):
                updates['Status'] = f"Assigned to {updates['Videographer']}"
                trello_updates_needed = True
                trello_updates['assignee'] = updates['Videographer']
            if 'Status' in updates and updates['Status'] != current_data.get('Status'):
                trello_updates_needed = True
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
        
        # Persist DB updates
        ok = db_update_task_by_number(task_number, updates)
        if not ok:
            return {"success": False, "error": "DB update failed"}
        
        # Trello updates
        if is_assigned and trello_updates_needed:
            trello_card = await get_trello_card_by_task_number(task_number)
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
                    lists = await get_trello_lists()
                    new_list_id = lists.get(trello_updates['assignee'])
                    if new_list_id:
                        trello_payload['idList'] = new_list_id
                if 'filming_date' in trello_updates:
                    from trello_utils import update_checklist_dates
                    await update_checklist_dates(trello_card['id'], trello_updates['filming_date'])
                if trello_payload:
                    success = await update_trello_card(trello_card['id'], trello_payload)
                    if not success:
                        return {"success": True, "warning": "DB updated but Trello update failed"}
            else:
                return {"success": True, "warning": "DB updated but Trello card not found"}
        
        return {"success": True, "updates": updates}
    except Exception as e:
        logger.error(f"Error updating task {task_number}: {e}")
        return {"success": False, "error": str(e)}
