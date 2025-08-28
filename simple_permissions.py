"""
Simple permissions system based on videographer_config.json
Supports group-based permissions with highest permission inheritance
"""

import json
from config import VIDEOGRAPHER_CONFIG_PATH

def load_config():
    """Load the videographer config"""
    try:
        with open(VIDEOGRAPHER_CONFIG_PATH, 'r') as f:
            return json.load(f)
    except:
        return {}

def get_user_groups(user_id: str) -> list:
    """Get all groups a user belongs to"""
    config = load_config()
    groups = []
    
    # Check all sections for user membership
    sections = {
        'super_admin': 'super_admin',
        'admin': 'admin', 
        'hod': 'hod',
        'head_of_sales': 'head_of_sales',
        'reviewer': 'reviewer',
        'videographers': 'videographers',
        'sales_people': 'sales_people'
    }
    
    for section, group_name in sections.items():
        if section in ['reviewer', 'hod', 'head_of_sales']:
            # Single person sections
            person = config.get(section, {})
            if person.get('slack_user_id') == user_id:
                groups.append(group_name)
        else:
            # Multi-person sections
            people = config.get(section, {})
            for person_name, person_data in people.items():
                if person_data.get('slack_user_id') == user_id:
                    groups.append(group_name)
                    break
    
    return groups

def get_user_permissions(user_id: str) -> set:
    """Get all permissions for a user based on their groups"""
    config = load_config()
    groups = get_user_groups(user_id)
    group_perms = config.get('group_permissions', {})
    
    # Collect all permissions from all groups
    all_permissions = set()
    for group in groups:
        if group in group_perms:
            all_permissions.update(group_perms[group])
    
    return all_permissions

def get_user_id_by_name(name: str) -> str:
    """Get Slack user ID by person's name"""
    config = load_config()
    
    # Check all sections
    for section in ['super_admin', 'admin', 'videographers', 'sales_people', 'reviewer', 'hod', 'head_of_sales']:
        if section in ['reviewer', 'hod', 'head_of_sales']:
            # Single person sections
            person = config.get(section, {})
            if person.get('name', '').lower() == name.lower():
                return person.get('slack_user_id', '')
        else:
            # Multi-person sections
            people = config.get(section, {})
            for person_name, person_data in people.items():
                if person_data.get('name', person_name).lower() == name.lower():
                    return person_data.get('slack_user_id', '')
    return ''

def can_user_do(user_id: str, action: str) -> bool:
    """Check if user can perform an action using group-based permissions"""
    config = load_config()
    
    # First check if it's available to everyone
    permissions = config.get('permissions', {})
    allowed_groups = permissions.get(action, [])
    
    if 'everyone' in [g.lower() for g in allowed_groups]:
        return True
    
    # Get user's permissions from all their groups
    user_permissions = get_user_permissions(user_id)
    
    # Check if the action is in user's permissions
    if action in user_permissions:
        return True
    
    # Also check legacy group keywords in permissions
    user_groups = get_user_groups(user_id)
    for group in user_groups:
        if group in allowed_groups:
            return True
    
    return False

def get_name_by_user_id(user_id: str) -> str:
    """Get person's name by their Slack user ID"""
    config = load_config()
    
    # Check all sections
    for section in ['super_admin', 'admin', 'videographers', 'sales_people', 'reviewer', 'hod', 'head_of_sales']:
        if section in ['reviewer', 'hod', 'head_of_sales']:
            # Single person sections
            person = config.get(section, {})
            if person.get('slack_user_id') == user_id:
                return person.get('name', section.replace('_', ' ').title())
        else:
            # Multi-person sections
            people = config.get(section, {})
            for person_name, person_data in people.items():
                if person_data.get('slack_user_id') == user_id:
                    return person_data.get('name', person_name)
    return 'Unknown'

# Simple check functions
def check_permission(user_id: str, action: str) -> tuple[bool, str]:
    """Check permission and return (allowed, error_message)"""
    if can_user_do(user_id, action):
        return True, ""
    
    user_name = get_name_by_user_id(user_id)
    return False, f"❌ {user_name}, you don't have permission to {action.replace('_', ' ')}"

# Example usage:
# In videographer_config.json, add:
# "permissions": {
#     "upload_task": ["Juliana Ribeiro", "Head of Sales", "everyone"],
#     "edit_task": ["Juliana Ribeiro", "Head of Sales"],
#     "delete_task": ["Juliana Ribeiro"],
#     "upload_video": ["videographers"],
#     "approve_video": ["Deaa Zararey", "Head of Sales"],
#     "manage_users": ["Juliana Ribeiro"]
# }