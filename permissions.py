"""
Permission management system for VideoCritique
Manages permissions for all Slack activities while maintaining compatibility with videographer_config.json
"""

import json
import os
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from config import VIDEOGRAPHER_CONFIG_PATH


class SlackActivity(Enum):
    """All available Slack activities in the system"""
    # Task management
    UPLOAD_TASK = "upload_task"  # Upload a task (text/pic)
    EDIT_TASK = "edit_task"
    DELETE_TASK = "delete_task"
    VIEW_EXCEL_DB = "view_excel_db"  # View excel and database
    
    # Video management
    UPLOAD_VIDEO = "upload_video"  # Videographer uploads video
    APPROVE_VIDEO_REVIEWER = "approve_video_reviewer"  # Reviewer approves video
    APPROVE_VIDEO_HOS = "approve_video_hos"  # Head of Sales approves video after reviewer
    RECEIVE_VIDEO_SALES = "receive_video_sales"  # Sales receives video
    
    # User/Location management
    ADD_SALES = "add_sales"
    LIST_SALES = "list_sales"
    DELETE_SALES = "delete_sales"
    ADD_VIDEOGRAPHER = "add_videographer"
    LIST_VIDEOGRAPHER = "list_videographer"
    DELETE_VIDEOGRAPHER = "delete_videographer"
    ADD_LOCATION_MAPPING = "add_location_mapping"
    LIST_LOCATION_MAPPING = "list_location_mapping"
    DELETE_LOCATION_MAPPING = "delete_location_mapping"
    
    # System management
    UPDATE_SLACK_IDS = "update_slack_ids"


@dataclass
class UserRole:
    """Define a user role with its allowed activities"""
    name: str
    description: str
    activities: Set[SlackActivity] = field(default_factory=set)
    
    def can_perform(self, activity: SlackActivity) -> bool:
        """Check if role can perform a specific activity"""
        return activity in self.activities


# Define roles with their allowed activities
ROLES = {
    "super_admin": UserRole(
        name="Super Admin",
        description="Full system access with all activities",
        activities=set(SlackActivity)  # All activities
    ),
    
    "admin": UserRole(
        name="Admin",
        description="Administrative access for task and user management",
        activities={
            # Task management
            SlackActivity.UPLOAD_TASK,
            SlackActivity.EDIT_TASK,
            SlackActivity.DELETE_TASK,
            SlackActivity.VIEW_EXCEL_DB,
            # User/Location management
            SlackActivity.ADD_SALES,
            SlackActivity.LIST_SALES,
            SlackActivity.DELETE_SALES,
            SlackActivity.ADD_VIDEOGRAPHER,
            SlackActivity.LIST_VIDEOGRAPHER,
            SlackActivity.DELETE_VIDEOGRAPHER,
            SlackActivity.ADD_LOCATION_MAPPING,
            SlackActivity.LIST_LOCATION_MAPPING,
            SlackActivity.DELETE_LOCATION_MAPPING,
            SlackActivity.UPDATE_SLACK_IDS,
        }
    ),
    
    "head_of_sales": UserRole(
        name="Head of Sales",
        description="Can manage tasks and do final video approval",
        activities={
            SlackActivity.UPLOAD_TASK,
            SlackActivity.EDIT_TASK,
            SlackActivity.VIEW_EXCEL_DB,
            SlackActivity.APPROVE_VIDEO_HOS,
            SlackActivity.LIST_SALES,
            SlackActivity.LIST_VIDEOGRAPHER,
            SlackActivity.LIST_LOCATION_MAPPING,
        }
    ),
    
    "reviewer": UserRole(
        name="Reviewer",
        description="Can review and approve/reject videos",
        activities={
            SlackActivity.VIEW_EXCEL_DB,
            SlackActivity.APPROVE_VIDEO_REVIEWER,
            SlackActivity.LIST_VIDEOGRAPHER,
            SlackActivity.LIST_LOCATION_MAPPING,
        }
    ),
    
    "videographer": UserRole(
        name="Videographer",
        description="Can upload videos and view their tasks",
        activities={
            SlackActivity.UPLOAD_VIDEO,
            SlackActivity.VIEW_EXCEL_DB,
        }
    ),
    
    "sales": UserRole(
        name="Sales",
        description="Can receive approved videos",
        activities={
            SlackActivity.RECEIVE_VIDEO_SALES,
            SlackActivity.VIEW_EXCEL_DB,
        }
    ),
}


def load_config():
    """Load the videographer config to get user mappings"""
    try:
        with open(VIDEOGRAPHER_CONFIG_PATH, 'r') as f:
            return json.load(f)
    except:
        return {}


def load_permission_config() -> Dict[str, str]:
    """Load permission config from file if exists"""
    from config import PERMISSIONS_CONFIG_PATH
    if os.path.exists(PERMISSIONS_CONFIG_PATH):
        try:
            with open(PERMISSIONS_CONFIG_PATH, 'r') as f:
                config = json.load(f)
                return config.get("user_roles", {})
        except:
            pass
    return {}


def load_user_info_from_config() -> Dict[str, Dict[str, str]]:
    """Load user info including both user IDs and channel IDs from config"""
    config = load_config()
    user_info = {}
    
    # Load reviewer
    reviewer = config.get("reviewer", {})
    if reviewer.get("slack_user_id"):
        user_info[reviewer["slack_user_id"]] = {
            "role": "reviewer",
            "channel_id": reviewer.get("slack_channel_id", ""),
            "name": reviewer.get("name", "Reviewer"),
            "email": reviewer.get("email", "")
        }
    
    # Load head of department
    hod = config.get("hod", {})
    if hod.get("slack_user_id"):
        user_info[hod["slack_user_id"]] = {
            "role": "admin",
            "channel_id": hod.get("slack_channel_id", ""),
            "name": hod.get("name", "Head of Department"),
            "email": hod.get("email", "")
        }
    
    # Load head of sales
    head_of_sales = config.get("head_of_sales", {})
    if head_of_sales.get("slack_user_id"):
        user_info[head_of_sales["slack_user_id"]] = {
            "role": "head_of_sales",
            "channel_id": head_of_sales.get("slack_channel_id", ""),
            "name": head_of_sales.get("name", "Head of Sales"),
            "email": head_of_sales.get("email", "")
        }
    
    # Load videographers
    videographers = config.get("videographers", {})
    for vg_name, vg_data in videographers.items():
        if vg_data.get("slack_user_id"):
            user_info[vg_data["slack_user_id"]] = {
                "role": "videographer",
                "channel_id": vg_data.get("slack_channel_id", ""),
                "name": vg_data.get("name", vg_name),
                "email": vg_data.get("email", "")
            }
    
    # Load sales people
    sales_list = config.get("sales_people", {})  # Note: it's sales_people in the actual config
    for sales_name, sales_data in sales_list.items():
        if sales_data.get("slack_user_id"):
            user_info[sales_data["slack_user_id"]] = {
                "role": "sales",
                "channel_id": sales_data.get("slack_channel_id", ""),
                "name": sales_data.get("name", sales_name),
                "email": sales_data.get("email", "")
            }
    
    # Load from permission config if exists (allows manual overrides)
    permission_config = load_permission_config()
    for user_id, role in permission_config.items():
        if user_id not in user_info:
            user_info[user_id] = {"role": role, "channel_id": "", "name": "", "email": ""}
        else:
            user_info[user_id]["role"] = role  # Override role from permission config
    
    return user_info


# User info including roles and channel IDs
USER_INFO: Dict[str, Dict[str, str]] = load_user_info_from_config()

# Extract just roles for backward compatibility
USER_ROLES: Dict[str, str] = {user_id: info["role"] for user_id, info in USER_INFO.items()}


class PermissionManager:
    """Manage user permissions and access control for Slack activities"""
    
    def __init__(self, user_info: Optional[Dict[str, Dict[str, str]]] = None):
        self.user_info = user_info or USER_INFO
        self.user_roles = {uid: info["role"] for uid, info in self.user_info.items()}
        self.roles = ROLES
    
    def get_user_role(self, user_id: str) -> Optional[UserRole]:
        """Get the role for a user"""
        role_name = self.user_roles.get(user_id)
        if role_name:
            return self.roles.get(role_name)
        return None
    
    def can_perform_activity(self, user_id: str, activity: SlackActivity) -> bool:
        """Check if a user can perform a specific activity"""
        role = self.get_user_role(user_id)
        if role:
            return role.can_perform(activity)
        return False
    
    def can_perform_any_activity(self, user_id: str, activities: List[SlackActivity]) -> bool:
        """Check if user can perform any of the specified activities"""
        role = self.get_user_role(user_id)
        if role:
            return any(role.can_perform(activity) for activity in activities)
        return False
    
    def get_user_activities(self, user_id: str) -> Set[SlackActivity]:
        """Get all activities a user can perform"""
        role = self.get_user_role(user_id)
        if role:
            return role.activities.copy()
        return set()
    
    def is_super_admin(self, user_id: str) -> bool:
        """Check if user is a super admin"""
        return self.user_roles.get(user_id) == "super_admin"
    
    def is_admin(self, user_id: str) -> bool:
        """Check if user is an admin or super admin"""
        role_name = self.user_roles.get(user_id)
        return role_name in ["admin", "super_admin"]
    
    def add_user(self, user_id: str, role_name: str) -> bool:
        """Add a user to a role"""
        if role_name in self.roles:
            self.user_roles[user_id] = role_name
            self.save_to_permission_config()
            return True
        return False
    
    def remove_user(self, user_id: str) -> bool:
        """Remove a user from the system"""
        if user_id in self.user_roles:
            del self.user_roles[user_id]
            self.save_to_permission_config()
            return True
        return False
    
    def update_user_role(self, user_id: str, new_role_name: str) -> bool:
        """Update a user's role"""
        if new_role_name in self.roles:
            self.user_roles[user_id] = new_role_name
            self.save_to_permission_config()
            return True
        return False
    
    def list_users_in_role(self, role_name: str) -> List[str]:
        """List all users in a specific role"""
        return [user_id for user_id, role in self.user_roles.items() 
                if role == role_name]
    
    def reload_from_config(self):
        """Reload user info from config files"""
        self.user_info = load_user_info_from_config()
        self.user_roles = {uid: info["role"] for uid, info in self.user_info.items()}
    
    def save_to_permission_config(self):
        """Save current user roles to permission config"""
        from config import PERMISSIONS_CONFIG_PATH
        config = {}
        
        # Load existing config if exists
        if os.path.exists(PERMISSIONS_CONFIG_PATH):
            try:
                with open(PERMISSIONS_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
            except:
                config = {}
        
        # Update user roles
        config["user_roles"] = self.user_roles
        
        # Save config
        try:
            with open(PERMISSIONS_CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving permission config: {e}")


# Create a global permission manager instance
permission_manager = PermissionManager()


def get_user_channel_id(user_id: str) -> Optional[str]:
    """Get the Slack channel ID for a user"""
    user_info = USER_INFO.get(user_id, {})
    return user_info.get("channel_id")


def get_user_info_by_role(role: str) -> List[Dict[str, str]]:
    """Get all users with a specific role"""
    users = []
    for uid, info in USER_INFO.items():
        if info.get("role") == role:
            users.append({"user_id": uid, **info})
    return users


def get_videographer_channel_id(videographer_name: str) -> Optional[str]:
    """Get channel ID for a videographer by name"""
    for uid, info in USER_INFO.items():
        if info.get("role") == "videographer" and info.get("name") == videographer_name:
            return info.get("channel_id")
    return None


def get_reviewer_info() -> Dict[str, str]:
    """Get reviewer info including user ID and channel ID"""
    for uid, info in USER_INFO.items():
        if info.get("role") == "reviewer":
            return {"user_id": uid, **info}
    return {}


def get_hod_info() -> Dict[str, str]:
    """Get head of department info"""
    for uid, info in USER_INFO.items():
        if info.get("role") == "admin":
            return {"user_id": uid, **info}
    return {}


def get_head_of_sales_info() -> Dict[str, str]:
    """Get head of sales info"""
    for uid, info in USER_INFO.items():
        if info.get("role") == "head_of_sales":
            return {"user_id": uid, **info}
    return {}


# Helper function to check permissions with error messages
def check_permission(user_id: str, activity: SlackActivity) -> tuple[bool, str]:
    """Check if user has permission and return appropriate message"""
    if permission_manager.can_perform_activity(user_id, activity):
        return True, ""
    
    role = permission_manager.get_user_role(user_id)
    if role:
        return False, f"❌ You don't have permission to {activity.value.replace('_', ' ')}. Your role ({role.name}) doesn't include this activity."
    else:
        return False, "❌ You are not authorized to use this system. Please contact an administrator."


# Backward compatibility - keep the old functions working
def get_admin_permissions():
    """Get admin user IDs for backward compatibility"""
    admin_ids = set()
    for user_id, role in permission_manager.user_roles.items():
        if role in ["admin", "super_admin"]:
            admin_ids.add(user_id)
    return admin_ids or {"U093E4CPEE4", "U088S1J93MW", "U087WM4H6SJ"}  # Fallback


# Legacy permission sets for backward compatibility
ADMIN_PERMISSIONS = get_admin_permissions()
CREATE_PERMISSIONS = get_admin_permissions()
EDIT_PERMISSIONS = get_admin_permissions()