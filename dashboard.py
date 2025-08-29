"""
Dashboard API implementation - clean and robust
Handles all dashboard data calculations and metrics
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
from fastapi.responses import JSONResponse

from config import UAE_TZ, HISTORY_DB_PATH
from db_utils import get_all_tasks_df
from logger import logger


async def get_historical_tasks_df() -> pd.DataFrame:
    """Get all completed tasks from history database as a DataFrame"""
    try:
        with sqlite3.connect(HISTORY_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT task_number, brand, campaign_start_date, campaign_end_date, 
                   reference_number, location, sales_person, submitted_by, status, 
                   filming_date, videographer, current_version, version_history,
                   pending_timestamps, submitted_timestamps, returned_timestamps,
                   rejected_timestamps, accepted_timestamps, completed_at
                FROM completed_tasks
                WHERE status != 'Archived'"""  # Exclude archived tasks
            )
            rows = cursor.fetchall()
            
            if rows:
                data = []
                for row in rows:
                    task_data = dict(row)
                    # Map to expected column names (same as live tasks)
                    mapped_data = {
                        'Task #': task_data['task_number'],
                        'Brand': task_data['brand'] or '',
                        'Campaign Start Date': task_data['campaign_start_date'] or '',
                        'Campaign End Date': task_data['campaign_end_date'] or '',
                        'Reference Number': task_data['reference_number'] or '',
                        'Location': task_data['location'] or '',
                        'Sales Person': task_data['sales_person'] or '',
                        'Submitted By': task_data['submitted_by'] or '',
                        'Status': task_data['status'] or 'Done',
                        'Filming Date': task_data['filming_date'] or '',
                        'Videographer': task_data['videographer'] or '',
                        'Video Filename': '',  # Not stored in history
                        'Current Version': task_data['current_version'] or '',
                        'Version History': task_data['version_history'] or '[]',
                        'Timestamp': task_data['completed_at'] or '',
                        'Pending Timestamps': task_data['pending_timestamps'] or '',
                        'Submitted Timestamps': task_data['submitted_timestamps'] or '',
                        'Returned Timestamps': task_data['returned_timestamps'] or '',
                        'Rejected Timestamps': task_data['rejected_timestamps'] or '',
                        'Accepted Timestamps': task_data['accepted_timestamps'] or ''
                    }
                    data.append(mapped_data)
                return pd.DataFrame(data)
            else:
                # Return empty DataFrame with proper columns
                return pd.DataFrame(columns=[
                    'Task #', 'Brand', 'Campaign Start Date', 'Campaign End Date',
                    'Reference Number', 'Location', 'Sales Person', 'Submitted By',
                    'Status', 'Filming Date', 'Videographer', 'Video Filename', 
                    'Current Version', 'Version History', 'Timestamp',
                    'Pending Timestamps', 'Submitted Timestamps', 'Returned Timestamps',
                    'Rejected Timestamps', 'Accepted Timestamps'
                ])
    except Exception as e:
        logger.error(f"Error loading historical tasks: {e}")
        return pd.DataFrame()


def get_empty_dashboard_response(mode: str, period: str) -> Dict[str, Any]:
    """Return empty dashboard response structure"""
    return {
        "mode": mode,
        "period": period,
        "pie": {"completed": 0, "not_completed": 0},
        "summary": {
            "total": 0,
            "assigned": 0,
            "pending": 0,
            "rejected": 0,
            "submitted_to_sales": 0,
            "returned": 0,
            "uploads": 0,
            "accepted_videos": 0,
            "accepted_pct": 0.0,
            "rejected_pct": 0.0
        },
        "reviewer": {
            "avg_response_hours": 0,
            "avg_response_display": "0 hrs",
            "pending_videos": 0,
            "handled": 0,
            "accepted": 0,
            "handled_percent": 0.0
        },
        "videographers": {},
        "summary_videographers": {}
    }


def safe_parse_date(date_str: Any) -> Optional[datetime]:
    """Safely parse date from various formats"""
    if pd.isna(date_str) or str(date_str).strip() == '':
        return None
    
    date_formats = ['%d-%m-%Y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']
    date_str = str(date_str).strip()
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except:
            continue
    
    try:
        return pd.to_datetime(date_str).date()
    except:
        return None


def is_date_in_period(date_val: Any, mode: str, period: str) -> bool:
    """Check if date falls within the specified period"""
    if date_val is None:
        return False
    
    try:
        if mode == 'year':
            return date_val.year == int(period)
        else:  # month mode
            year, month = map(int, period.split('-'))
            return date_val.year == year and date_val.month == month
    except:
        return False


def safe_parse_json(json_str: Any, default: Any = None) -> Any:
    """Safely parse JSON string"""
    if default is None:
        default = []
    
    if pd.isna(json_str) or str(json_str).strip() in ['', 'nan', 'None']:
        return default
    
    try:
        result = json.loads(str(json_str))
        return result if isinstance(result, list) else default
    except:
        return default


def safe_str(value: Any) -> str:
    """Convert any value to string safely"""
    if pd.isna(value) or value is None:
        return ''
    return str(value)


def calculate_percentage(numerator: float, denominator: float) -> float:
    """Calculate percentage safely"""
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 1)


def format_duration(hours: float) -> str:
    """Format duration in hours to human readable string"""
    if hours <= 0:
        return "0 hrs"
    
    if hours < 1:
        return f"{round(hours * 60)} mins"
    elif hours < 24:
        return f"{round(hours, 1)} hrs"
    else:
        days = int(hours / 24)
        remaining_hours = round(hours % 24, 1)
        if remaining_hours > 0:
            return f"{days}d {remaining_hours}h"
        return f"{days}d"


async def get_history_count_in_period(mode: str, period: str) -> int:
    """Get count of completed tasks from history database in the specified period"""
    try:
        with sqlite3.connect(HISTORY_DB_PATH) as conn:
            if mode == 'year':
                # Match any date ending with the year (DD-MM-YYYY)
                pattern = f"%-{period}"
            else:  # month mode, period like YYYY-MM
                year, month = period.split('-')
                # Match DD-MM-YYYY format: any day in this month/year
                pattern = f"%-{month}-{year}"
            
            cursor = conn.execute(
                "SELECT COUNT(*) FROM completed_tasks WHERE filming_date LIKE ? AND status != 'Archived'",
                (pattern,)
            )
            return cursor.fetchone()[0]
    except Exception as e:
        logger.warning(f"Error querying history database: {e}")
        return 0


async def api_dashboard(mode: str = "month", period: str = ""):
    """Main dashboard API endpoint"""
    try:
        logger.info(f"\n=== DASHBOARD REQUEST: mode={mode}, period={period} ===")
        
        # Default period to current if not provided
        if not period:
            period = datetime.now(UAE_TZ).strftime('%Y-%m' if mode == 'month' else '%Y')
        
        # Load live tasks data
        try:
            df = await get_all_tasks_df()
            logger.info(f"Loaded {len(df)} live tasks from database")
        except Exception as e:
            logger.error(f"Failed to read live tasks: {e}")
            return JSONResponse(get_empty_dashboard_response(mode, period))
        
        # Load historical tasks data
        try:
            history_df = await get_historical_tasks_df()
            logger.info(f"Loaded {len(history_df)} historical tasks from database")
            
            # Combine live and historical tasks
            if len(history_df) > 0:
                df = pd.concat([df, history_df], ignore_index=True)
                logger.info(f"Total tasks after combining: {len(df)}")
        except Exception as e:
            logger.warning(f"Failed to read historical tasks: {e}")
            # Continue with just live tasks
        
        # Check if we have any data
        if len(df) == 0:
            logger.info("No tasks in Excel")
            return JSONResponse(get_empty_dashboard_response(mode, period))
        
        # Ensure required columns exist
        required_columns = {
            'Task #': '',
            'Brand': '',
            'Reference Number': '',
            'Filming Date': '',
            'Videographer': '',
            'Status': '',
            'Current Version': '',
            'Version History': '[]',
            'Campaign Start Date': '',
            'Campaign End Date': '',
            'Location': ''
        }
        
        for col, default in required_columns.items():
            if col not in df.columns:
                df[col] = default
        
        # Parse filming dates
        df['parsed_filming_date'] = df['Filming Date'].apply(safe_parse_date)
        
        # Filter tasks by period
        df['in_period'] = df['parsed_filming_date'].apply(
            lambda d: is_date_in_period(d, mode, period)
        )
        tasks_in_period = df[df['in_period']].copy()
        
        logger.info(f"Tasks in period: {len(tasks_in_period)}")
        
        # Return empty if no tasks in period
        if len(tasks_in_period) == 0:
            return JSONResponse(get_empty_dashboard_response(mode, period))
        
        # Initialize counters
        total_tasks = len(tasks_in_period)
        assigned_tasks = len(tasks_in_period[
            tasks_in_period['Videographer'].notna() & 
            (tasks_in_period['Videographer'] != '')
        ])
        
        # Process version history for metrics
        uploads = 0
        rejected_events = 0
        returned_events = 0
        accepted_events = 0
        submitted_events = 0
        pending_count = 0  # Track unique pending versions
        completed_tasks = 0
        
        # Process each task in period
        for idx, task in tasks_in_period.iterrows():
            version_history = safe_parse_json(task.get('Version History', '[]'))
            
            unique_versions = set()
            task_accepted = False
            
            for event in version_history:
                if not isinstance(event, dict):
                    continue
                
                folder = str(event.get('folder', '')).lower()
                version = event.get('version')
                
                # Count unique uploads
                if folder == 'pending' and version is not None:
                    unique_versions.add(version)
                    pending_count = len(unique_versions)  # Update pending count
                
                # Count events
                if folder == 'rejected':
                    rejected_events += 1
                elif folder == 'returned':
                    returned_events += 1
                elif folder == 'accepted':
                    accepted_events += 1
                    task_accepted = True
                elif folder == 'submitted to sales':
                    submitted_events += 1
            
            uploads += len(unique_versions)
            if task_accepted:
                completed_tasks += 1
        
        # Calculate totals
        total_completed = completed_tasks
        not_completed = max(total_tasks - total_completed, 0)
        
        # Calculate percentages with benefit of the doubt
        # For acceptance: count accepted + submitted to sales events as positive
        # Exclude currently pending videos from the denominator
        # Get current status counts (from ALL tasks, not filtered)
        all_statuses = df['Status'].fillna('').astype(str)
        current_pending = int((all_statuses == 'Critique').sum())
        current_submitted = int((all_statuses == 'Submitted to Sales').sum())
        
        decided_uploads = max(uploads - current_pending, 0)
        positive_outcomes = accepted_events + current_submitted
        
        accepted_pct = calculate_percentage(positive_outcomes, decided_uploads) if decided_uploads > 0 else 0.0
        rejected_pct = calculate_percentage(rejected_events + returned_events, decided_uploads) if decided_uploads > 0 else 0.0
        
        # Calculate reviewer metrics
        reviewer_stats = calculate_reviewer_stats(tasks_in_period)
        reviewer_stats['pending_videos'] = current_pending
        
        # Calculate videographer metrics
        videographer_data, videographer_summary = calculate_videographer_stats(
            tasks_in_period, df
        )
        
        # Build response
        response = {
            "mode": mode,
            "period": period,
            "pie": {
                "completed": total_completed,
                "not_completed": not_completed
            },
            "summary": {
                "total": total_tasks,
                "assigned": assigned_tasks,
                "pending": current_pending,
                "rejected": rejected_events,
                "submitted_to_sales": current_submitted,
                "returned": returned_events,
                "uploads": uploads,
                "accepted_videos": accepted_events,
                "accepted_pct": accepted_pct,
                "rejected_pct": rejected_pct
            },
            "reviewer": reviewer_stats,
            "videographers": videographer_data,
            "summary_videographers": videographer_summary
        }
        
        return JSONResponse(response)
        
    except Exception as e:
        logger.error(f"Dashboard API error: {e}", exc_info=True)
        return JSONResponse(
            get_empty_dashboard_response(mode, period), 
            status_code=500
        )


def calculate_reviewer_stats(tasks_in_period: pd.DataFrame) -> Dict[str, Any]:
    """Calculate reviewer statistics from tasks"""
    response_times = []
    reviewer_handled = 0
    submitted_to_accepted = 0  # Videos that went from submitted to accepted
    total_submitted = 0  # Total videos submitted to sales
    
    for idx, task in tasks_in_period.iterrows():
        version_history = safe_parse_json(task.get('Version History', '[]'))
        
        # Group events by version
        versions = {}
        for event in version_history:
            if not isinstance(event, dict):
                continue
            
            version = event.get('version')
            if version:
                if version not in versions:
                    versions[version] = []
                versions[version].append(event)
        
        # Analyze each version's journey
        for version, events in versions.items():
            pending_time = None
            was_submitted = False
            was_accepted = False
            
            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda x: x.get('at', ''))
            
            for event in sorted_events:
                folder = str(event.get('folder', '')).lower()
                timestamp_str = event.get('at', '')
                
                if not timestamp_str:
                    continue
                
                try:
                    timestamp = datetime.strptime(timestamp_str, '%d-%m-%Y %H:%M:%S')
                except:
                    continue
                
                if folder == 'pending' and pending_time is None:
                    pending_time = timestamp
                elif folder in ['rejected', 'submitted to sales'] and pending_time:
                    reviewer_handled += 1
                    
                    # Calculate response time
                    delta_hours = (timestamp - pending_time).total_seconds() / 3600.0
                    if delta_hours > 0:
                        response_times.append(delta_hours)
                
                # Track submission and acceptance
                if folder == 'submitted to sales' and not was_submitted:
                    was_submitted = True
                    total_submitted += 1
                elif folder == 'accepted' and was_submitted and not was_accepted:
                    was_accepted = True
                    submitted_to_accepted += 1
    
    # Calculate averages
    avg_hours = 0
    if response_times:
        avg_hours = sum(response_times) / len(response_times)  # Don't round yet to preserve small values
    
    handled_percent = calculate_percentage(submitted_to_accepted, total_submitted)
    
    return {
        "avg_response_hours": avg_hours,
        "avg_response_display": format_duration(avg_hours),
        "pending_videos": 0,  # Will be set by caller
        "handled": reviewer_handled,
        "accepted": submitted_to_accepted,
        "handled_percent": handled_percent,
        "total_submitted": total_submitted
    }


def calculate_videographer_stats(
    tasks_in_period: pd.DataFrame, 
    all_tasks: pd.DataFrame
) -> tuple[Dict[str, List], Dict[str, Dict]]:
    """Calculate per-videographer statistics"""
    videographer_data = {}
    videographer_summary = {}
    
    # Get unique videographers from tasks in period
    videographers = tasks_in_period[
        tasks_in_period['Videographer'].notna() & 
        (tasks_in_period['Videographer'] != '')
    ]['Videographer'].unique()
    
    for vg in videographers:
        vg_tasks = tasks_in_period[tasks_in_period['Videographer'] == vg]
        
        vg_data = []
        vg_uploads = 0
        vg_rejected = 0
        vg_returned = 0
        vg_accepted = 0
        vg_submitted_events = 0  # Track submitted to sales events
        
        # Process each task
        for idx, task in vg_tasks.iterrows():
            # Parse version history JSON
            version_history_str = task.get('Version History', '[]')
            version_history = safe_parse_json(version_history_str)
            
            # Initialize metadata
            filming_deadline = safe_str(task.get('Filming Date'))
            uploaded_version = 'NA'
            version_number = 'NA'
            submitted_at = 'NA'
            accepted_at = 'NA'
            
            # Process version history to extract latest events
            unique_versions = set()
            versions_dict = {}  # Group events by version number
            
            for event in version_history:
                if isinstance(event, dict):
                    folder = event.get('folder', '')
                    version = event.get('version')
                    timestamp = event.get('at', '')
                    
                    # Group events by version
                    if version:
                        if version not in versions_dict:
                            versions_dict[version] = {'version': version, 'lifecycle': []}
                        
                        # Create lifecycle event
                        lifecycle_event = {
                            'stage': folder,
                            'at': timestamp
                        }
                        
                        # Add rejection details if present
                        if folder.lower() in ['rejected', 'returned']:
                            if event.get('rejection_class'):
                                lifecycle_event['rejection_class'] = event['rejection_class']
                            if event.get('rejection_comments'):
                                lifecycle_event['rejection_comments'] = event['rejection_comments']
                            if event.get('rejected_by'):
                                lifecycle_event['rejected_by'] = event['rejected_by']
                        
                        versions_dict[version]['lifecycle'].append(lifecycle_event)
                    
                    # Track metadata based on folder type
                    if folder.lower() == 'pending' and version:
                        unique_versions.add(version)
                        # Update upload info (keep latest)
                        uploaded_version = timestamp
                        version_number = f"v{version}"
                    elif folder.lower() == 'submitted to sales':
                        submitted_at = timestamp
                        vg_submitted_events += 1  # Count submitted events
                    elif folder.lower() == 'accepted':
                        accepted_at = timestamp
                    
                    # Count events
                    if folder.lower() == 'rejected':
                        vg_rejected += 1
                    elif folder.lower() == 'returned':
                        vg_returned += 1
                    elif folder.lower() == 'accepted':
                        vg_accepted += 1
            
            # Convert versions dict to list sorted by version number (descending)
            versions_for_display = sorted(versions_dict.values(), key=lambda x: x['version'], reverse=True)
            
            # Build task info
            task_info = {
                'task_number': safe_str(task.get('Task #')),
                'brand': safe_str(task.get('Brand')),
                'reference': safe_str(task.get('Reference Number')),
                'status': safe_str(task.get('Status')),
                'filming_deadline': filming_deadline,
                'uploaded_version': uploaded_version,
                'version_number': version_number,
                'submitted_at': submitted_at,
                'accepted_at': accepted_at,
                'versions': versions_for_display  # Use the properly formatted version list
            }
            
            vg_uploads += len(unique_versions)
            vg_data.append(task_info)
        
        # Get current status counts (from ALL tasks)
        all_vg_tasks = all_tasks[all_tasks['Videographer'] == vg]
        vg_statuses = all_vg_tasks['Status'].fillna('').astype(str)
        vg_pending = int((vg_statuses == 'Critique').sum())
        vg_submitted = int((vg_statuses == 'Submitted to Sales').sum())
        
        videographer_data[vg] = vg_data
        
        # Calculate acceptance rate with benefit of the doubt
        # Consider accepted + currently submitted to sales as positive outcomes
        # Don't count currently pending videos against them
        positive_outcomes = vg_accepted + vg_submitted
        # For decided videos: total uploads minus currently pending
        decided_videos = vg_uploads - vg_pending
        acceptance_rate = calculate_percentage(positive_outcomes, decided_videos) if decided_videos > 0 else 100.0
        
        videographer_summary[vg] = {
            'total': len(vg_tasks),
            'uploads': vg_uploads,
            'pending': vg_pending,
            'rejected': vg_rejected,
            'submitted_to_sales': vg_submitted,
            'returned': vg_returned,
            'accepted_videos': vg_accepted,
            'accepted_pct': acceptance_rate
        }
    
    return videographer_data, videographer_summary