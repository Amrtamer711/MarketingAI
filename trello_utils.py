import requests
from typing import Dict, Any
from config import TRELLO_API_KEY, TRELLO_API_TOKEN, BOARD_NAME, VIDEOGRAPHER_CONFIG_PATH
from logger import logger
from config import WEEKEND_DAYS
from datetime import datetime, timedelta
from uae_holidays import is_working_day as is_uae_working_day, add_working_days as add_uae_working_days, count_working_days as count_uae_working_days
# Removed circular imports - these will be handled differently

# ========== TRELLO HELPER FUNCTIONS ==========
def get_trello_card_by_task_number(task_number: int):
    """Get Trello card by task number"""
    try:
        logger.debug(f"Searching for Trello card for Task #{task_number}")
        
        # Get board ID
        url = "https://api.trello.com/1/members/me/boards"
        params = {
            "key": TRELLO_API_KEY,
            "token": TRELLO_API_TOKEN
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        boards = response.json()
        
        board_id = None
        for board in boards:
            if board["name"] == BOARD_NAME:
                board_id = board["id"]
                logger.debug(f"Found board '{BOARD_NAME}' with ID: {board_id}")
                break
        
        if not board_id:
            logger.error(f"Board '{BOARD_NAME}' not found among {len(boards)} boards")
            return None
        
        # Get all cards on the board
        url = f"https://api.trello.com/1/boards/{board_id}/cards"
        response = requests.get(url, params=params)
        response.raise_for_status()
        cards = response.json()
        logger.debug(f"Found {len(cards)} total cards on board")
        
        # Search for card with task number in title
        search_text = f"Task #{task_number}:"
        for card in cards:
            if search_text in card['name']:
                logger.debug(f"Found matching card: {card['name']}")
                return card
        
        logger.debug(f"No card found with '{search_text}' in title")
        return None
        
    except Exception as e:
        logger.error(f"Error getting Trello card for Task #{task_number}: {e}")
        return None


def update_trello_card(card_id: str, updates: Dict[str, Any]):
    """Update a Trello card with new information"""
    try:
        url = f"https://api.trello.com/1/cards/{card_id}"
        query = {
            "key": TRELLO_API_KEY,
            "token": TRELLO_API_TOKEN
        }
        
        # Merge updates with query
        query.update(updates)
        
        response = requests.put(url, params=query)
        response.raise_for_status()
        
        return True
    except Exception as e:
        logger.error(f"Error updating Trello card: {e}")
        return False


def get_trello_lists():
    """Get all lists on the board"""
    try:
        # Get board ID first
        url = "https://api.trello.com/1/members/me/boards"
        params = {
            "key": TRELLO_API_KEY,
            "token": TRELLO_API_TOKEN
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        boards = response.json()
        
        board_id = None
        for board in boards:
            if board["name"] == BOARD_NAME:
                board_id = board["id"]
                break
        
        if not board_id:
            return {}
        
        # Get lists
        url = f"https://api.trello.com/1/boards/{board_id}/lists"
        response = requests.get(url, params=params)
        response.raise_for_status()
        lists = response.json()
        
        # Create mapping of person names to list IDs
        list_mapping = {}
        for lst in lists:
            list_mapping[lst["name"]] = lst["id"]
        
        return list_mapping
    except Exception as e:
        logger.error(f"Error getting Trello lists: {e}")
        return {}

def create_trello_list(board_id: str, list_name: str, position: str = "bottom"):
    """Create a new list on a Trello board"""
    try:
        url = f"https://api.trello.com/1/boards/{board_id}/lists"
        params = {
            "key": TRELLO_API_KEY,
            "token": TRELLO_API_TOKEN,
            "name": list_name,
            "pos": position
        }
        response = requests.post(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error creating Trello list: {e}")
        return None

def archive_trello_list(list_id: str):
    """Archive a Trello list"""
    try:
        url = f"https://api.trello.com/1/lists/{list_id}/closed"
        params = {
            "key": TRELLO_API_KEY,
            "token": TRELLO_API_TOKEN,
            "value": "true"
        }
        response = requests.put(url, params=params)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Error archiving Trello list: {e}")
        return False

def get_list_id_for_videographer(board_id: str, videographer_name: str):
    """Get Trello list ID for a videographer"""
    try:
        url = f"https://api.trello.com/1/boards/{board_id}/lists"
        params = {
            "key": TRELLO_API_KEY,
            "token": TRELLO_API_TOKEN
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        lists = response.json()
        
        for lst in lists:
            if lst["name"] == videographer_name and not lst.get("closed", False):
                return lst["id"]
        return None
    except Exception as e:
        logger.error(f"Error getting list ID: {e}")
        return None

def set_trello_due_complete(card_id: str, complete: bool) -> bool:
    """Set the Trello card's dueComplete checkbox (True/False)."""
    try:
        url = f"https://api.trello.com/1/cards/{card_id}"
        params = {
            "key": TRELLO_API_KEY,
            "token": TRELLO_API_TOKEN,
            "dueComplete": "true" if complete else "false"
        }
        resp = requests.put(url, params=params)
        resp.raise_for_status()
        print(f"   🔄 Trello dueComplete -> {complete}")
        return True
    except Exception as e:
        print(f"   ⚠️ Failed to set dueComplete: {e}")
        return False

def archive_trello_card(card_id: str) -> bool:
    """Archive the Trello card (remove from board view)."""
    try:
        url = f"https://api.trello.com/1/cards/{card_id}"
        params = {
            "key": TRELLO_API_KEY,
            "token": TRELLO_API_TOKEN,
            "closed": "true"
        }
        resp = requests.put(url, params=params)
        resp.raise_for_status()
        logger.info(f"🗄️ Archived Trello card {card_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to archive Trello card {card_id}: {e}")
        return False

def create_checklist_with_dates(card_id: str, filming_date: datetime) -> bool:
    """Create a checklist on a Trello card with filming and editing due dates"""
    try:
        # Create the checklist
        url = f"https://api.trello.com/1/checklists"
        params = {
            "key": TRELLO_API_KEY,
            "token": TRELLO_API_TOKEN,
            "idCard": card_id,
            "name": "Video Production Timeline"
        }
        response = requests.post(url, params=params)
        response.raise_for_status()
        checklist = response.json()
        checklist_id = checklist['id']
        
        # Add filming task with due date
        filming_date_str = filming_date.strftime('%Y-%m-%dT12:00:00.000Z')
        url = f"https://api.trello.com/1/checklists/{checklist_id}/checkItems"
        params = {
            "key": TRELLO_API_KEY,
            "token": TRELLO_API_TOKEN,
            "name": f"Filming - Due: {filming_date.strftime('%d-%m-%Y')}",
            "pos": "bottom",
            "due": filming_date_str
        }
        response = requests.post(url, params=params)
        response.raise_for_status()
        
        # Add editing task (3 working days after filming)
        editing_date = add_working_days(filming_date, 3)
        editing_date_str = editing_date.strftime('%Y-%m-%dT12:00:00.000Z')
        params = {
            "key": TRELLO_API_KEY,
            "token": TRELLO_API_TOKEN,
            "name": f"Editing - Due: {editing_date.strftime('%d-%m-%Y')}",
            "pos": "bottom",
            "due": editing_date_str
        }
        response = requests.post(url, params=params)
        response.raise_for_status()
        
        return True
    except Exception as e:
        logger.error(f"Error creating checklist: {e}")
        return False

def update_checklist_dates(card_id: str, new_filming_date: datetime) -> bool:
    """Update checklist dates when filming date changes"""
    try:
        # Get all checklists on the card
        url = f"https://api.trello.com/1/cards/{card_id}/checklists"
        params = {
            "key": TRELLO_API_KEY,
            "token": TRELLO_API_TOKEN
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        checklists = response.json()
        
        # Find the production timeline checklist
        for checklist in checklists:
            if checklist['name'] == 'Video Production Timeline':
                # Get checklist items
                url = f"https://api.trello.com/1/checklists/{checklist['id']}/checkItems"
                response = requests.get(url, params=params)
                response.raise_for_status()
                items = response.json()
                
                # Update filming date item
                for item in items:
                    if 'Filming' in item['name']:
                        new_name = f"Filming - Due: {new_filming_date.strftime('%d-%m-%Y')}"
                        url = f"https://api.trello.com/1/cards/{card_id}/checkItem/{item['id']}"
                        update_params = params.copy()
                        update_params['name'] = new_name
                        update_params['due'] = new_filming_date.strftime('%Y-%m-%dT12:00:00.000Z')
                        response = requests.put(url, params=update_params)
                        response.raise_for_status()
                    
                    elif 'Editing' in item['name']:
                        # Update editing date (3 working days after new filming date)
                        editing_date = add_working_days(new_filming_date, 3)
                        new_name = f"Editing - Due: {editing_date.strftime('%d-%m-%Y')}"
                        url = f"https://api.trello.com/1/cards/{card_id}/checkItem/{item['id']}"
                        update_params = params.copy()
                        update_params['name'] = new_name
                        update_params['due'] = editing_date.strftime('%Y-%m-%dT12:00:00.000Z')
                        response = requests.put(url, params=update_params)
                        response.raise_for_status()
                
                return True
        
        return False
    except Exception as e:
        logger.error(f"Error updating checklist dates: {e}")
        return False

def is_weekend(date: datetime) -> bool:
    return date.weekday() in WEEKEND_DAYS

def is_on_leave(name: str, date: datetime) -> bool:
    # For now, just check if it's a weekend
    # In the future, this could check actual leave schedules
    return is_weekend(date)

def add_working_days(start_date: datetime, num_days: int) -> datetime:
    """Add working days to a date, skipping weekends and UAE holidays"""
    # Convert datetime to date for uae_holidays functions
    if isinstance(start_date, datetime):
        start_date_obj = start_date.date()
    else:
        start_date_obj = start_date
    
    # Use holiday-aware function
    result_date = add_uae_working_days(start_date_obj, num_days, holiday_pad_days=0)
    
    # Convert back to datetime if needed
    if isinstance(start_date, datetime):
        return datetime.combine(result_date, start_date.time())
    return result_date

def count_working_days(start_date: datetime, end_date: datetime) -> int:
    """Count working days between two dates (inclusive of start, exclusive of end), accounting for weekends and holidays"""
    # Convert to date objects
    if isinstance(start_date, datetime):
        start_date_obj = start_date.date()
    else:
        start_date_obj = start_date
        
    if isinstance(end_date, datetime):
        end_date_obj = end_date.date()
    else:
        end_date_obj = end_date
    
    # Adjust for exclusive end date
    if end_date_obj > start_date_obj:
        end_date_obj = end_date_obj - timedelta(days=1)
    
    # Use holiday-aware function
    return count_uae_working_days(start_date_obj, end_date_obj, holiday_pad_days=0)

def calculate_filming_date(campaign_start_date: datetime, campaign_end_date: datetime = None) -> datetime:
    """Calculate filming date based on simplified rules:
    1. Main filming dates are 4th and 17th of each month
    2. If campaign is too short and won't land on those days, do 2 days post campaign start
    3. If filming lands on non-working day, use next working day
    4. But if campaign will end by then, use working day before
    """
    # Convert dates to datetime if needed
    import datetime as dt
    if isinstance(campaign_start_date, dt.date) and not isinstance(campaign_start_date, datetime):
        campaign_start_date = datetime.combine(campaign_start_date, datetime.min.time())
    if campaign_end_date and isinstance(campaign_end_date, dt.date) and not isinstance(campaign_end_date, datetime):
        campaign_end_date = datetime.combine(campaign_end_date, datetime.min.time())
    
    # Find the next filming date (4th or 17th)
    current_date = campaign_start_date
    current_day = current_date.day
    current_month = current_date.month
    current_year = current_date.year
    
    # Determine next filming date
    if current_day <= 4:
        # Next filming date is 4th of current month
        next_filming = datetime(current_year, current_month, 4)
    elif current_day <= 17:
        # Next filming date is 17th of current month
        next_filming = datetime(current_year, current_month, 17)
    else:
        # Next filming date is 4th of next month
        if current_month == 12:
            next_filming = datetime(current_year + 1, 1, 4)
        else:
            next_filming = datetime(current_year, current_month + 1, 4)
    
    # Check if campaign ends before the filming date (end date is inclusive)
    if campaign_end_date and next_filming.date() > campaign_end_date.date():
        # Campaign too short - use 2 working days post campaign start
        return add_working_days(campaign_start_date, 2)
    
    # Check if filming date is a working day
    if not is_uae_working_day(next_filming.date(), holiday_pad_days=0):
        # Find nearest working day (check both directions)
        days_forward = 0
        days_backward = 0
        
        # Find next working day
        next_working = next_filming
        while not is_uae_working_day(next_working.date(), holiday_pad_days=0):
            next_working += timedelta(days=1)
            days_forward += 1
            if days_forward > 30:  # Safety limit
                break
        
        # Find previous working day
        prev_working = next_filming
        while not is_uae_working_day(prev_working.date(), holiday_pad_days=0):
            prev_working -= timedelta(days=1)
            days_backward += 1
            if days_backward > 30:  # Safety limit
                break
        
        # Choose nearest working day
        if days_forward <= days_backward:
            # Next working day is closer or equal distance
            if campaign_end_date and next_working.date() > campaign_end_date.date():
                # But if it exceeds campaign end (which is inclusive), use previous
                return prev_working
            return next_working
        else:
            # Previous working day is closer
            return prev_working
    
    return next_filming

def get_assigned_videographer(location: str) -> str:
    return LOCATION_TO_VIDEOGRAPHER.get(location, "Unassigned")

# Load configuration directly to avoid circular import
import json

def _load_videographer_config():
    """Load videographer configuration from JSON file"""
    config_path = VIDEOGRAPHER_CONFIG_PATH
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading videographer config: {e}")
        return {"videographers": {}, "location_mappings": {}}

videographer_config = _load_videographer_config()
LOCATION_TO_VIDEOGRAPHER = videographer_config["location_mappings"]
ALL_VIDEOGRAPHERS = list(videographer_config["videographers"].keys())

# ========== PLATFORM INTEGRATION FUNCTIONS ==========

def get_videographer_cards(board_id, videographer_name):
    """Get all cards assigned to a videographer on their list"""
    try:
        # Get the videographer's list
        list_id = get_list_id_by_name(board_id, videographer_name)
        if not list_id:
            return []
        
        # Get all cards in the list
        url = f"https://api.trello.com/1/lists/{list_id}/cards"
        response = requests.get(url, 
                                params = {
                                    "key": TRELLO_API_KEY,
                                    "token": TRELLO_API_TOKEN
                                })
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting cards for {videographer_name}: {e}")
        return []

def is_videographer_on_leave(board_id, videographer_name):
    """Check if videographer is on leave (has an 'On Leave' card)"""
    cards = get_videographer_cards(board_id, videographer_name)
    
    # Check if any card is titled "On Leave"
    for card in cards:
        if card.get("name", "").lower() == "on leave":
            return True
    return False

def get_leave_dates(board_id, videographer_name):
    """Get the start and end dates from the 'On Leave' card if it exists
    Returns tuple of (start_date, end_date) or (None, None)
    
    Note: Trello cards support both 'start' and 'due' dates.
    We'll use 'start' for leave start date and 'due' for leave end date.
    If 'start' is not set, we'll fall back to card creation date.
    """
    cards = get_videographer_cards(board_id, videographer_name)
    
    # Find the "On Leave" card
    for card in cards:
        if card.get("name", "").lower() == "on leave":
            # Start date - prefer 'start' field, fallback to creation date
            start_date_str = card.get("start")
            if not start_date_str:
                # Use creation date as fallback
                start_date_str = card.get("dateLastActivity")
            
            # End date is the due date
            end_date_str = card.get("due")
            
            start_date = None
            end_date = None
            
            if start_date_str:
                try:
                    start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00')).date()
                except:
                    pass
                    
            if end_date_str:
                try:
                    end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')).date()
                except:
                    pass
                    
            return start_date, end_date
    
    return None, None

def get_leave_end_date(board_id, videographer_name):
    """Get the due date from the 'On Leave' card if it exists"""
    _, end_date = get_leave_dates(board_id, videographer_name)
    return end_date

def will_be_available_by_date(board_id, videographer_name, target_date):
    """Check if videographer will be available on the target date (not on leave)
    
    Simple logic: If filming date falls between leave start and end (inclusive), 
    the person is NOT available.
    """
    if not is_videographer_on_leave(board_id, videographer_name):
        return True  # Not on leave, so available
    
    start_date, end_date = get_leave_dates(board_id, videographer_name)
    
    # Convert target_date to date object if it's datetime
    if isinstance(target_date, datetime):
        target_date = target_date.date()
    
    # If no dates specified, assume unavailable
    if not start_date and not end_date:
        return False
    
    # If only end date specified, assume on leave until that date (inclusive)
    if not start_date and end_date:
        return target_date > end_date
    
    # If only start date specified, assume on leave from that date onwards
    if start_date and not end_date:
        return target_date < start_date
    
    # Both dates specified - check if filming falls within leave period (inclusive)
    # Person is NOT available if: start_date <= target_date <= end_date
    return not (start_date <= target_date <= end_date)

def count_videographer_workload(board_id, videographer_name):
    """Count the number of active tasks for a videographer"""
    cards = get_videographer_cards(board_id, videographer_name)
    
    # Filter out "On Leave" cards
    active_cards = [card for card in cards if card.get("name", "").lower() != "on leave"]
    return len(active_cards)

def get_all_workloads(board_id, exclude_on_leave=True, target_date=None):
    """Get workload for all videographers, optionally checking who will be available by target_date"""
    workloads = {}
    on_leave = []
    
    for videographer in ALL_VIDEOGRAPHERS:
        # Check if on leave
        if target_date and not will_be_available_by_date(board_id, videographer, target_date):
            on_leave.append(videographer)
            if exclude_on_leave:
                continue
        elif not target_date and is_videographer_on_leave(board_id, videographer):
            on_leave.append(videographer)
            if exclude_on_leave:
                continue
        
        # Count workload
        workload = count_videographer_workload(board_id, videographer)
        workloads[videographer] = workload
    
    return workloads, on_leave

def get_board_id_by_name(board_name):
    """Get Trello board ID by name"""
    url = "https://api.trello.com/1/members/me/boards"
    response = requests.get(url, 
                            params = {
                                "key": TRELLO_API_KEY,
                                "token": TRELLO_API_TOKEN
                            })
    response.raise_for_status()
    boards = response.json()
    for board in boards:
        if board["name"] == board_name:
            return board["id"]
    raise Exception(f"Board '{board_name}' not found")

def get_list_id_by_name(board_id, list_name):
    """Get Trello list ID by name"""
    url = f"https://api.trello.com/1/boards/{board_id}/lists"
    response = requests.get(url, 
                            params = {
                                "key": TRELLO_API_KEY,
                                "token": TRELLO_API_TOKEN
                            })
    response.raise_for_status()
    lists = response.json()
    
    for l in lists:
        if l["name"].lower() == list_name.lower():
            return l["id"]
    return lists[0]["id"] if lists else None

def create_card_on_board(board_name, card_title, card_description, due_date=None, list_name="Test", start_date=None):
    """Create a card on Trello board"""
    board_id = get_board_id_by_name(board_name)
    list_id = get_list_id_by_name(board_id, list_name)
    
    if not list_id:
        raise Exception(f"No list found in board '{board_name}'")

    url = "https://api.trello.com/1/cards"
    payload = {
        "key": TRELLO_API_KEY,
        "token": TRELLO_API_TOKEN,
        "idList": list_id,
        "name": card_title,
        "desc": card_description,
        "pos": "bottom"
    }
    if due_date:
        payload["due"] = due_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    if start_date:
        payload["start"] = start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    response = requests.post(url, params=payload)
    response.raise_for_status()
    card = response.json()
    return card

def get_best_videographer_for_balancing(board_id, primary_person=None, reason="leave", target_date=None):
    """Get the best videographer based on the balancing reason
    
    Args:
        board_id: Trello board ID
        primary_person: Person to exclude from selection
        reason: Reason for balancing (currently unused, reserved for future use)
        target_date: Date to check availability
    """
    workloads, on_leave = get_all_workloads(board_id, target_date=target_date)
    
    # Remove primary person and those on leave from consideration
    available_workloads = {k: v for k, v in workloads.items() 
                          if k != primary_person and k not in on_leave}
    
    if not available_workloads:
        return None, {}
    
    # Find videographer with minimum workload
    best_videographer = min(available_workloads, key=available_workloads.get)
    
    # Print workload summary
    print(f"  📊 Workload Summary:")
    for person, load in sorted(workloads.items(), key=lambda x: x[1]):
        status = ""
        if person in on_leave:
            status = " (ON LEAVE)"
        elif person == primary_person:
            status = " (PRIMARY)"
        elif person == best_videographer:
            status = " (SELECTED)"
        print(f"     • {person}: {load} tasks{status}")
    
    return best_videographer, workloads
