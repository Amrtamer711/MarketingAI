import json
from config import VIDEOGRAPHER_CONFIG_PATH

def load_config():
    """Load the videographer config to get admin user IDs"""
    config_path = VIDEOGRAPHER_CONFIG_PATH
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def get_admin_permissions():
    """Get admin permissions from reviewer, hod, and head_of_sales in config"""
    config = load_config()
    admin_ids = set()
    
    # Add reviewer's Slack user ID if exists
    reviewer = config.get("reviewer", {})
    if reviewer.get("slack_user_id"):
        admin_ids.add(reviewer["slack_user_id"])
    
    # Add head of department's Slack user ID if exists
    hod = config.get("hod", {})
    if hod.get("slack_user_id"):
        admin_ids.add(hod["slack_user_id"])
    
    # Add head of sales's Slack user ID if exists
    head_of_sales = config.get("head_of_sales", {})
    if head_of_sales.get("slack_user_id"):
        admin_ids.add(head_of_sales["slack_user_id"])
    
    # Fallback to hardcoded IDs if no admin IDs found in config
    if not admin_ids:
        admin_ids = {"U093E4CPEE4", "U088S1J93MW", "U087WM4H6SJ"}  # Fallback admin IDs
    
    return admin_ids

# Permission lists
# To get a user's Slack ID:
# 1. In Slack, click on the user's profile
# 2. Click "More" (three dots)
# 3. Click "Copy member ID"

# For now, all permissions are the same until we implement role-based access
CREATE_PERMISSIONS = {
    "U093E4CPEE4", "U088S1J93MW", "U087WM4H6SJ"  # Example user IDs - replace with actual IDs
}

EDIT_PERMISSIONS = {
    "U093E4CPEE4", "U088S1J93MW", "U087WM4H6SJ"  # Example user IDs - replace with actual IDs
}

# Admin permissions dynamically loaded from config (reviewer and hod only)
ADMIN_PERMISSIONS = get_admin_permissions()