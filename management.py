import json
from pathlib import Path
from typing import Dict, Any
from config import TRELLO_API_KEY, TRELLO_API_TOKEN, BOARD_NAME, VIDEOGRAPHER_CONFIG_PATH
from logger import logger
from trello_utils import create_trello_list, archive_trello_list, get_list_id_for_videographer
from db_utils import get_all_tasks_df
import re
import requests

# ========== VIDEOGRAPHER MANAGEMENT FUNCTIONS ==========
def load_videographer_config():
    """Load videographer configuration from JSON file"""
    config_path = VIDEOGRAPHER_CONFIG_PATH
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading videographer config: {e}")
        return {"videographers": {}, "location_mappings": {}}

def save_videographer_config(config):
    """Save videographer configuration to JSON file"""
    config_path = VIDEOGRAPHER_CONFIG_PATH
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving videographer config: {e}")
        return False



async def add_videographer(name: str, email: str, slack_user_id: str = "", slack_channel_id: str = "") -> Dict[str, Any]:
    """Add a new videographer to the system with complete profile"""
    try:
        # Load current config
        config = load_videographer_config()
        
        # Check if videographer already exists
        if name in config["videographers"]:
            return {"success": False, "error": f"Videographer '{name}' already exists"}
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return {"success": False, "error": "Invalid email format"}
        
        # Get Trello board ID
        try:
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
                return {"success": False, "error": f"Trello board '{BOARD_NAME}' not found"}
            
            # Create Trello list for the videographer
            trello_list = await create_trello_list(board_id, name)
            if not trello_list:
                return {"success": False, "error": "Failed to create Trello list"}
            
        except Exception as e:
            return {"success": False, "error": f"Trello error: {str(e)}"}
        
        # Add to config with complete profile
        config["videographers"][name] = {
            "name": name,
            "email": email,
            "slack_user_id": slack_user_id or "",
            "slack_channel_id": slack_channel_id or "",
            "active": True
        }
        
        # Save config
        if not save_videographer_config(config):
            # Rollback Trello list creation
            await archive_trello_list(trello_list["id"])
            return {"success": False, "error": "Failed to save configuration"}
        
        return {
            "success": True,
            "message": f"Videographer '{name}' added successfully",
            "trello_list_id": trello_list["id"]
        }
        
    except Exception as e:
        logger.error(f"Error adding videographer: {e}")
        return {"success": False, "error": str(e)}

async def remove_videographer(name: str) -> Dict[str, Any]:
    """Remove a videographer from the system"""
    try:
        # Load current config
        config = load_videographer_config()
        
        # Check if videographer exists
        if name not in config["videographers"]:
            return {"success": False, "error": f"Videographer '{name}' not found"}
        
        # Check if videographer has assigned locations
        assigned_locations = [loc for loc, vid in config["location_mappings"].items() if vid == name]
        if assigned_locations:
            return {
                "success": False,
                "error": f"Cannot remove '{name}' - has assigned locations: {', '.join(assigned_locations)}",
                "assigned_locations": assigned_locations
            }
        
        # Check for active tasks in Excel
        try:
            df = await get_all_tasks_df()
            active_tasks = df[(df['Videographer'] == name) & (df['Status'].str.startswith('Assigned to'))]
            if len(active_tasks) > 0:
                return {
                    "success": False,
                    "error": f"Cannot remove '{name}' - has {len(active_tasks)} active tasks",
                    "active_tasks": len(active_tasks)
                }
        except Exception as e:
            logger.warning(f"Could not check Excel for active tasks: {e}")
        
        # Archive Trello list
        try:
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
            
            if board_id:
                list_id = await get_list_id_for_videographer(board_id, name)
                if list_id:
                    await archive_trello_list(list_id)
        except Exception as e:
            logger.warning(f"Could not archive Trello list: {e}")
        
        # Remove from config
        del config["videographers"][name]
        
        # Save config
        if not save_videographer_config(config):
            return {"success": False, "error": "Failed to save configuration"}
        
        return {
            "success": True,
            "message": f"Videographer '{name}' removed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error removing videographer: {e}")
        return {"success": False, "error": str(e)}

async def add_location(location: str, videographer: str) -> Dict[str, Any]:
    """Add a location mapping to a videographer"""
    try:
        # Load current config
        config = load_videographer_config()
        
        # Check if videographer exists
        if videographer not in config["videographers"]:
            return {"success": False, "error": f"Videographer '{videographer}' not found"}
        
        # Check if location already mapped
        if location in config["location_mappings"]:
            current = config["location_mappings"][location]
            if current == videographer:
                return {"success": False, "error": f"Location '{location}' already assigned to '{videographer}'"}
            else:
                return {"success": False, "error": f"Location '{location}' already assigned to '{current}'"}
        
        # Add mapping
        config["location_mappings"][location] = videographer
        
        # Save config
        if not save_videographer_config(config):
            return {"success": False, "error": "Failed to save configuration"}
        
        return {
            "success": True,
            "message": f"Location '{location}' mapped to '{videographer}'"
        }
        
    except Exception as e:
        logger.error(f"Error adding location: {e}")
        return {"success": False, "error": str(e)}

async def remove_location(location: str) -> Dict[str, Any]:
    """Remove a location mapping"""
    try:
        # Load current config
        config = load_videographer_config()
        
        # Check if location exists
        if location not in config["location_mappings"]:
            return {"success": False, "error": f"Location '{location}' not found"}
        
        # Check for tasks with this location in Excel
        try:
            df = await get_all_tasks_df()
            location_tasks = df[(df['Location'] == location) & (df['Status'] == 'Not assigned yet')]
            if len(location_tasks) > 0:
                return {
                    "success": False,
                    "error": f"Cannot remove '{location}' - has {len(location_tasks)} unassigned tasks",
                    "unassigned_tasks": len(location_tasks)
                }
        except Exception as e:
            logger.warning(f"Could not check Excel for location tasks: {e}")
        
        # Remove mapping
        videographer = config["location_mappings"][location]
        del config["location_mappings"][location]
        
        # Save config
        if not save_videographer_config(config):
            return {"success": False, "error": "Failed to save configuration"}
        
        return {
            "success": True,
            "message": f"Location '{location}' unmapped from '{videographer}'"
        }
        
    except Exception as e:
        logger.error(f"Error removing location: {e}")
        return {"success": False, "error": str(e)}

async def list_videographers() -> Dict[str, Any]:
    """List all videographers and their details"""
    try:
        config = load_videographer_config()
        
        # Count assigned locations per videographer
        location_counts = {}
        for videographer in config["videographers"]:
            location_counts[videographer] = sum(1 for v in config["location_mappings"].values() if v == videographer)
        
        return {
            "success": True,
            "videographers": config["videographers"],
            "location_counts": location_counts,
            "total_videographers": len(config["videographers"])
        }
    except Exception as e:
        logger.error(f"Error listing videographers: {e}")
        return {"success": False, "error": str(e)}

async def list_locations() -> Dict[str, Any]:
    """List all location mappings"""
    try:
        config = load_videographer_config()
        
        return {
            "success": True,
            "location_mappings": config["location_mappings"],
            "total_locations": len(config["location_mappings"])
        }
    except Exception as e:
        logger.error(f"Error listing locations: {e}")
        return {"success": False, "error": str(e)}

# ========== SALESPERSON MANAGEMENT FUNCTIONS ==========
async def add_salesperson(name: str, email: str, slack_user_id: str = "", slack_channel_id: str = "") -> Dict[str, Any]:
    """Add a new salesperson to the system with complete profile"""
    try:
        # Load current config
        config = load_videographer_config()
        
        # Check if salesperson already exists
        if name in config.get("sales_people", {}):
            return {"success": False, "error": f"Salesperson '{name}' already exists"}
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return {"success": False, "error": "Invalid email format"}
        
        # Add to config
        if "sales_people" not in config:
            config["sales_people"] = {}
        
        config["sales_people"][name] = {
            "name": name,
            "email": email,
            "slack_user_id": slack_user_id or "",
            "slack_channel_id": slack_channel_id or "",
            "active": True
        }
        
        # Save config
        if not save_videographer_config(config):
            return {"success": False, "error": "Failed to save configuration"}
        
        return {
            "success": True,
            "message": f"Salesperson '{name}' added successfully"
        }
        
    except Exception as e:
        logger.error(f"Error adding salesperson: {e}")
        return {"success": False, "error": str(e)}

async def remove_salesperson(name: str) -> Dict[str, Any]:
    """Remove a salesperson from the system"""
    try:
        # Load current config
        config = load_videographer_config()
        
        # Check if salesperson exists
        if name not in config.get("sales_people", {}):
            return {"success": False, "error": f"Salesperson '{name}' not found"}
        
        # Check for active tasks with this salesperson in Excel
        try:
            df = await get_all_tasks_df()
            active_tasks = df[(df['Sales Person'] == name) & (df['Status'] != 'Done')]
            if len(active_tasks) > 0:
                return {
                    "success": False,
                    "error": f"Cannot remove '{name}' - has {len(active_tasks)} active tasks",
                    "active_tasks": len(active_tasks)
                }
        except Exception as e:
            logger.warning(f"Could not check Excel for salesperson tasks: {e}")
        
        # Remove from config
        del config["sales_people"][name]
        
        # Save config
        if not save_videographer_config(config):
            return {"success": False, "error": "Failed to save configuration"}
        
        return {
            "success": True,
            "message": f"Salesperson '{name}' removed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error removing salesperson: {e}")
        return {"success": False, "error": str(e)}

async def list_salespeople() -> Dict[str, Any]:
    """List all salespeople and their details"""
    try:
        config = load_videographer_config()
        
        # Count active tasks per salesperson
        task_counts = {}
        try:
            df = await get_all_tasks_df()
            for salesperson in config.get("sales_people", {}):
                active_tasks = df[(df['Sales Person'] == salesperson) & (df['Status'] != 'Done')]
                task_counts[salesperson] = len(active_tasks)
        except Exception as e:
            logger.warning(f"Could not count salesperson tasks: {e}")
        
        return {
            "success": True,
            "salespeople": config.get("sales_people", {}),
            "task_counts": task_counts,
            "total_salespeople": len(config.get("sales_people", {}))
        }
    except Exception as e:
        logger.error(f"Error listing salespeople: {e}")
        return {"success": False, "error": str(e)}

async def update_person_slack_ids(person_type: str, person_name: str, slack_user_id: str = None, slack_channel_id: str = None) -> Dict[str, Any]:
    """Update Slack IDs for any person in the system"""
    try:
        config = load_videographer_config()
        
        # Find the person
        if person_type == "reviewer":
            if "reviewer" not in config:
                config["reviewer"] = {
                    "name": "Reviewer",
                    "email": "",
                    "slack_user_id": "",
                    "slack_channel_id": "",
                    "active": True
                }
            person = config["reviewer"]
            person_ref = "Reviewer"
        elif person_type == "hod":
            if "hod" not in config:
                config["hod"] = {
                    "name": "Head of Department",
                    "email": "",
                    "slack_user_id": "",
                    "slack_channel_id": "",
                    "active": True
                }
            person = config["hod"]
            person_ref = "Head of Department"
        elif person_type == "head_of_sales":
            if "head_of_sales" not in config:
                config["head_of_sales"] = {
                    "name": "Head of Sales",
                    "email": "",
                    "slack_user_id": "",
                    "slack_channel_id": "",
                    "active": True
                }
            person = config["head_of_sales"]
            person_ref = "Head of Sales"
        else:
            # videographers or sales_people
            people = config.get(person_type, {})
            if person_name not in people:
                return {"success": False, "error": f"{person_name} not found in {person_type}"}
            person = people[person_name]
            person_ref = person_name
        
        # Update IDs if provided
        updated = False
        if slack_user_id is not None:
            person["slack_user_id"] = slack_user_id
            updated = True
        if slack_channel_id is not None:
            person["slack_channel_id"] = slack_channel_id
            updated = True
        
        if not updated:
            return {"success": False, "error": "No IDs provided to update"}
        
        # Save config
        if not save_videographer_config(config):
            return {"success": False, "error": "Failed to save configuration"}
        
        return {
            "success": True,
            "message": f"Updated Slack IDs for {person_ref}"
        }
        
    except Exception as e:
        logger.error(f"Error updating Slack IDs: {e}")
        return {"success": False, "error": str(e)}

# ========== SINGLE-PERSON STAKEHOLDER EDIT FUNCTIONS ==========
async def edit_reviewer(name: str = None, email: str = None, slack_user_id: str = None, slack_channel_id: str = None, active: bool = None) -> Dict[str, Any]:
    """Edit the reviewer's information"""
    try:
        config = load_videographer_config()
        
        # Initialize reviewer if not exists
        if "reviewer" not in config:
            config["reviewer"] = {
                "name": "Reviewer",
                "email": "",
                "slack_user_id": "",
                "slack_channel_id": "",
                "active": True
            }
        
        reviewer = config["reviewer"]
        updated_fields = []
        
        # Update provided fields
        if name is not None:
            reviewer["name"] = name
            updated_fields.append(f"Name: {name}")
        if email is not None:
            # Validate email format
            if email:  # Only validate if not empty
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, email):
                    return {"success": False, "error": "Invalid email format"}
            reviewer["email"] = email
            updated_fields.append(f"Email: {email}")
        if slack_user_id is not None:
            reviewer["slack_user_id"] = slack_user_id
            updated_fields.append(f"Slack User ID: {slack_user_id}")
        if slack_channel_id is not None:
            reviewer["slack_channel_id"] = slack_channel_id
            updated_fields.append(f"Slack Channel ID: {slack_channel_id}")
        if active is not None:
            reviewer["active"] = active
            updated_fields.append(f"Active: {active}")
        
        if not updated_fields:
            return {"success": False, "error": "No fields provided to update"}
        
        # Save config
        if not save_videographer_config(config):
            return {"success": False, "error": "Failed to save configuration"}
        
        return {
            "success": True,
            "message": f"Updated reviewer information",
            "updated_fields": updated_fields,
            "reviewer": reviewer
        }
        
    except Exception as e:
        logger.error(f"Error editing reviewer: {e}")
        return {"success": False, "error": str(e)}

async def edit_hod(name: str = None, email: str = None, slack_user_id: str = None, slack_channel_id: str = None, active: bool = None) -> Dict[str, Any]:
    """Edit the Head of Department's information"""
    try:
        config = load_videographer_config()
        
        # Initialize HOD if not exists
        if "hod" not in config:
            config["hod"] = {
                "name": "Head of Department",
                "email": "",
                "slack_user_id": "",
                "slack_channel_id": "",
                "active": True
            }
        
        hod = config["hod"]
        updated_fields = []
        
        # Update provided fields
        if name is not None:
            hod["name"] = name
            updated_fields.append(f"Name: {name}")
        if email is not None:
            # Validate email format
            if email:  # Only validate if not empty
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, email):
                    return {"success": False, "error": "Invalid email format"}
            hod["email"] = email
            updated_fields.append(f"Email: {email}")
        if slack_user_id is not None:
            hod["slack_user_id"] = slack_user_id
            updated_fields.append(f"Slack User ID: {slack_user_id}")
        if slack_channel_id is not None:
            hod["slack_channel_id"] = slack_channel_id
            updated_fields.append(f"Slack Channel ID: {slack_channel_id}")
        if active is not None:
            hod["active"] = active
            updated_fields.append(f"Active: {active}")
        
        if not updated_fields:
            return {"success": False, "error": "No fields provided to update"}
        
        # Save config
        if not save_videographer_config(config):
            return {"success": False, "error": "Failed to save configuration"}
        
        return {
            "success": True,
            "message": f"Updated Head of Department information",
            "updated_fields": updated_fields,
            "hod": hod
        }
        
    except Exception as e:
        logger.error(f"Error editing HOD: {e}")
        return {"success": False, "error": str(e)}

async def edit_head_of_sales(name: str = None, email: str = None, slack_user_id: str = None, slack_channel_id: str = None, active: bool = None) -> Dict[str, Any]:
    """Edit the Head of Sales' information"""
    try:
        config = load_videographer_config()
        
        # Initialize Head of Sales if not exists
        if "head_of_sales" not in config:
            config["head_of_sales"] = {
                "name": "Head of Sales",
                "email": "",
                "slack_user_id": "",
                "slack_channel_id": "",
                "active": True
            }
        
        hos = config["head_of_sales"]
        updated_fields = []
        
        # Update provided fields
        if name is not None:
            hos["name"] = name
            updated_fields.append(f"Name: {name}")
        if email is not None:
            # Validate email format
            if email:  # Only validate if not empty
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, email):
                    return {"success": False, "error": "Invalid email format"}
            hos["email"] = email
            updated_fields.append(f"Email: {email}")
        if slack_user_id is not None:
            hos["slack_user_id"] = slack_user_id
            updated_fields.append(f"Slack User ID: {slack_user_id}")
        if slack_channel_id is not None:
            hos["slack_channel_id"] = slack_channel_id
            updated_fields.append(f"Slack Channel ID: {slack_channel_id}")
        if active is not None:
            hos["active"] = active
            updated_fields.append(f"Active: {active}")
        
        if not updated_fields:
            return {"success": False, "error": "No fields provided to update"}
        
        # Save config
        if not save_videographer_config(config):
            return {"success": False, "error": "Failed to save configuration"}
        
        return {
            "success": True,
            "message": f"Updated Head of Sales information",
            "updated_fields": updated_fields,
            "head_of_sales": hos
        }
        
    except Exception as e:
        logger.error(f"Error editing Head of Sales: {e}")
        return {"success": False, "error": str(e)}

