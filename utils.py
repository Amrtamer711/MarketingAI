import json
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd
from uae_holidays import add_working_days, is_working_day, get_next_working_day, get_previous_working_day
from logger import logger
# Note: user_history will be imported where needed to avoid circular imports
import re


__all__ = [
	"EmailParseRequest",
	"RequestFilter",
	"_load_mapping_config",
	"_format_sales_people_hint",
	"_format_locations_hint",
	"_format_videographers_hint",
	"append_to_history",
]

# ========== PYDANTIC MODELS ==========
class EmailParseRequest(BaseModel):
	email_text: str
	save_to_excel: bool = False
	submitted_by: str = "API"

class RequestFilter(BaseModel):
	start_date: Optional[str] = None
	end_date: Optional[str] = None
	brand: Optional[str] = None


# ===== Dynamic mapping helpers (module scope) =====
def _load_mapping_config() -> Dict[str, Any]:
	try:
		from config import VIDEOGRAPHER_CONFIG_PATH
		with open(VIDEOGRAPHER_CONFIG_PATH, "r") as f:
			return json.load(f)
	except Exception:
		return {"videographers": {}, "sales_people": {}, "location_mappings": {}}

def _format_sales_people_hint(cfg: Dict[str, Any]) -> str:
	try:
		items = []
		for name in cfg.get("sales_people", {}):
			parts = name.split()
			if parts:
				code = (parts[0][:1] + (parts[-1] if len(parts) > 1 else "")).upper()
				items.append(f"{code}→{name}")
		return ", ".join(items)
	except Exception:
		return ""

def _format_locations_hint(cfg: Dict[str, Any]) -> str:
	try:
		return ", ".join(cfg.get("location_mappings", {}).keys())
	except Exception:
		return ""

def _format_videographers_hint(cfg: Dict[str, Any]) -> str:
	try:
		return ", ".join(cfg.get("videographers", {}).keys())
	except Exception:
		return ""

# ========== UTILITY FUNCTIONS ==========
def add_working_days(start_date, days, holiday_pad_days=0):
    """Add working days to a date (excluding weekends and holidays)"""
    # Use the imported function from uae_holidays which handles holidays
    from uae_holidays import add_working_days as uae_add_working_days
    return uae_add_working_days(start_date, days, holiday_pad_days)

def is_weekend(date_obj) -> bool:
    """Check if a date falls on weekend (Friday=4, Saturday=5, Sunday=6 in UAE)"""
    if hasattr(date_obj, 'weekday'):
        return date_obj.weekday() in [4, 5, 6]  # Friday, Saturday and Sunday
    return False

# Note: For full working day checks including holidays, use is_working_day() from uae_holidays module

def calculate_filming_date(campaign_start_date_str: str, campaign_end_date_str: str = None) -> str:
    """Calculate filming date based on rules:
    1. Main filming dates are 4th and 17th of each month
    2. Move all requests to the closest NEXT filming date (4th or 17th)
    3. If campaign is short and ends before next filming date, do 2 working days after campaign start
    4. If filming date (4th/17th) falls on non-working day (holiday/weekend), use next closest working day
    5. If campaign ends by then, use previous closest working day
    
    Args:
        campaign_start_date_str: Campaign start date in dd-mm-yyyy format
        campaign_end_date_str: Campaign end date in dd-mm-yyyy format (optional)
    
    Returns:
        Filming date in dd-mm-yyyy format
    """
    try:
        # Parse dates - handle dd-mm-yyyy format
        if '-' in campaign_start_date_str and len(campaign_start_date_str.split('-')[0]) == 2:
            # Already in dd-mm-yyyy format
            campaign_start_date = pd.to_datetime(campaign_start_date_str, format='%d-%m-%Y').date()
        else:
            # Try pandas default parsing
            campaign_start_date = pd.to_datetime(campaign_start_date_str).date()
        
        campaign_end_date = None
        if campaign_end_date_str:
            try:
                if '-' in campaign_end_date_str and len(campaign_end_date_str.split('-')[0]) == 2:
                    campaign_end_date = pd.to_datetime(campaign_end_date_str, format='%d-%m-%Y').date()
                else:
                    campaign_end_date = pd.to_datetime(campaign_end_date_str).date()
            except:
                logger.warning(f"Could not parse campaign end date: {campaign_end_date_str}")
        
        # Find the next filming date (4th or 17th) AFTER campaign start
        current_day = campaign_start_date.day
        current_month = campaign_start_date.month
        current_year = campaign_start_date.year
        
        # Determine next filming date
        if current_day < 4:
            # Next filming date is 4th of current month
            next_filming = datetime(current_year, current_month, 4).date()
        elif current_day < 17:
            # Next filming date is 17th of current month
            next_filming = datetime(current_year, current_month, 17).date()
        else:
            # Next filming date is 4th of next month
            if current_month == 12:
                next_filming = datetime(current_year + 1, 1, 4).date()
            else:
                next_filming = datetime(current_year, current_month + 1, 4).date()
        
        # IMPORTANT: Ensure filming date is AFTER campaign start
        # If campaign starts on 4th or 17th, still push to next filming date
        if next_filming <= campaign_start_date:
            if next_filming.day == 4:
                # Move to 17th of same month
                next_filming = datetime(current_year, current_month, 17).date()
            else:
                # Move to 4th of next month
                if current_month == 12:
                    next_filming = datetime(current_year + 1, 1, 4).date()
                else:
                    next_filming = datetime(current_year, current_month + 1, 4).date()
        
        # Check if campaign is short and ends before the next filming date
        if campaign_end_date and next_filming > campaign_end_date:
            # Use fallback: 2 working days after campaign start
            filming_date = add_working_days(campaign_start_date, 2, holiday_pad_days=0)
            logger.info(f"Campaign ends {campaign_end_date}, before next filming date {next_filming}. Using 2 working days after start: {filming_date}")
            return filming_date.strftime("%d-%m-%Y")
        
        # Check if the filming date is a working day
        # Use holiday_pad_days=0 to only consider actual holiday day (no padding)
        if is_working_day(next_filming, holiday_pad_days=0):
            # It's a working day, use it as is
            return next_filming.strftime("%d-%m-%Y")
        
        # Filming date falls on non-working day, need to adjust
        # First try next working day
        next_working = get_next_working_day(next_filming, holiday_pad_days=0)
        
        # Check if campaign would end before the next working day
        if campaign_end_date and next_working > campaign_end_date:
            # Try previous working day instead
            prev_working = get_previous_working_day(next_filming, holiday_pad_days=0)
            
            # Make sure previous working day is after campaign start
            if prev_working > campaign_start_date:
                logger.info(f"Filming date {next_filming} is non-working, next working {next_working} exceeds campaign end. Using previous working day: {prev_working}")
                return prev_working.strftime("%d-%m-%Y")
            else:
                # Can't use previous, fall back to 2 working days after start
                filming_date = add_working_days(campaign_start_date, 2, holiday_pad_days=0)
                logger.info(f"No suitable working day found around {next_filming}. Using 2 working days after start: {filming_date}")
                return filming_date.strftime("%d-%m-%Y")
        else:
            # Use next working day
            logger.info(f"Filming date {next_filming} is non-working. Using next working day: {next_working}")
            return next_working.strftime("%d-%m-%Y")
        
    except Exception as e:
        logger.error(f"Error calculating filming date: {e}")
        # Return empty string on error
        return ""

def markdown_to_slack(text: str) -> str:
    """Convert markdown formatting to Slack formatting"""
    # Horizontal rules: --- -> divider (remove them as Slack doesn't have equivalent)
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    
    # Headers: # Header -> *Header*
    text = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
    
    # Bold: **text** or __text__ -> *text*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    text = re.sub(r'__(.+?)__', r'*\1*', text)
    
    # Italic: *text* or _text_ -> _text_
    # First temporarily replace ** patterns to avoid conflicts
    text = re.sub(r'\*\*', '§§', text)
    text = re.sub(r'\*(.+?)\*', r'_\1_', text)
    text = re.sub('§§', '**', text)
    
    # Code blocks: ```code``` -> ```code```
    # Slack uses the same format for code blocks
    
    # Inline code: `code` -> `code`
    # Slack uses the same format for inline code
    
    # Links: [text](url) -> <url|text>
    text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<\2|\1>', text)
    
    # Bullet points: - item -> • item
    text = re.sub(r'^-\s+', '• ', text, flags=re.MULTILINE)
    
    # Numbered lists: 1. item -> 1. item (same format)
    
    # Blockquotes: > text -> > text (same format)
    
    # Strikethrough: ~~text~~ -> ~text~
    text = re.sub(r'~~(.+?)~~', r'~\1~', text)
    
    # Clean up extra newlines that might result from removing horizontal rules
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def append_to_history(user_id: str, role: str, content: str):
    """Add message to user's conversation history"""
    # This function will be called with user_history passed from the caller
    # to avoid circular imports
    from history import user_history
    
    if user_id not in user_history:
        user_history[user_id] = []
    
    user_history[user_id].append({"role": role, "content": content})
    
    # Keep only last 20 messages to avoid token limits
    if len(user_history[user_id]) > 20:
        user_history[user_id] = user_history[user_id][-20:]

# Note: add_working_days is now imported from uae_holidays module
# It handles both weekends (Friday/Saturday in UAE) and holidays

def check_time_conditions(filming_date, current_date=None):
    """Check if it's past 12 PM the next working day after filming"""
    today = current_date if current_date else datetime.now()
    filming_datetime = pd.to_datetime(filming_date).to_pydatetime()
    
    # Calculate next working day after filming (using holiday-aware function)
    filming_date_only = filming_datetime.date()
    next_working_day_date = add_working_days(filming_date_only, 1)
    next_working_day = datetime.combine(next_working_day_date, filming_datetime.time())
    
    # Check if we're past the next working day
    if today.date() < next_working_day.date():
        return False, False, 0  # Not yet time to check
    
    # Check if it's past 12 PM
    noon_deadline = next_working_day.replace(hour=12, minute=0, second=0)
    five_pm_deadline = next_working_day.replace(hour=17, minute=0, second=0)
    
    past_noon = today >= noon_deadline
    past_five_pm = today >= five_pm_deadline
    
    # Calculate working days overdue (holiday-aware)
    working_days_overdue = 0
    check_date = filming_datetime.date()
    while check_date < today.date():
        if is_working_day(check_date):
            working_days_overdue += 1
        check_date += timedelta(days=1)
    
    return past_noon, past_five_pm, working_days_overdue

def check_raw_folder_deadline(filming_date, current_date=None):
    """Check if video in raw folder is overdue (must move within 3 working days of submission)
    Returns: (is_overdue, working_days_in_raw)
    """
    today = current_date if current_date else datetime.now()
    filming_datetime = pd.to_datetime(filming_date).to_pydatetime()
    
    # Submission deadline is 1 working day after filming (by 5 PM)
    filming_date_only = filming_datetime.date()
    submission_date_only = add_working_days(filming_date_only, 1)
    submission_date = datetime.combine(submission_date_only, filming_datetime.time())
    
    # Video must move from raw within 3 working days after submission
    # So total of 4 working days after filming
    move_deadline_date = add_working_days(filming_date_only, 4)
    move_deadline = datetime.combine(move_deadline_date, filming_datetime.time())
    
    # Check if we're past the move deadline
    is_overdue = today.date() > move_deadline.date()
    
    # Calculate working days since submission (holiday-aware)
    working_days_in_raw = 0
    check_date = submission_date.date()
    while check_date < today.date():
        if is_working_day(check_date):
            working_days_in_raw += 1
        check_date += timedelta(days=1)
    
    return is_overdue, working_days_in_raw