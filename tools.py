# Define available tools
functions = [
    {
        "type": "function",
        "name": "log_design_request",
        "description": "Log a new design request with specific details.",
        "parameters": {
            "type": "object",
            "properties": {
                "brand": {"type": "string", "description": "Brand or client name"},
                "campaign_start_date": {"type": "string", "description": "Campaign start date in YYYY-MM-DD format"},
                "campaign_end_date": {"type": "string", "description": "Campaign end date in YYYY-MM-DD format"},
                "reference_number": {"type": "string", "description": "Reference number"},
                "location": {"type": "string", "description": "Campaign location (required)"},
                "sales_person": {"type": "string", "description": "Sales person name (required)"}
            },
            "required": ["brand", "campaign_start_date", "campaign_end_date", "reference_number", "location", "sales_person"]
        }
    },
    {
        "type": "function",
        "name": "export_current_data",
        "description": "Send the actual Excel file and history database files to the user via Slack. This exports the raw data files for download and external viewing.",
        "parameters": {
            "type": "object",
            "properties": {
                "include_history": {"type": "boolean", "description": "Include the history database file with completed tasks", "default": True}
            }
        }
    },
    {
        "type": "function",
        "name": "edit_task",
        "description": "Edit or view details of an existing task by task number. Use this whenever the user mentions a specific task number they want to edit, update, modify, or change. This shows current task details before allowing edits.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_number": {"type": "integer", "description": "The task number to edit"}
            },
            "required": ["task_number"]
        }
    },
    {
        "type": "function",
        "name": "manage_videographer",
        "description": "Manage videographers in the system - add, remove, or list videographers",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "remove", "list"],
                    "description": "The action to perform"
                },
                "name": {
                    "type": "string",
                    "description": "Videographer name (required for add/remove)"
                },
                "email": {
                    "type": "string",
                    "description": "Videographer email (required for add)"
                },
                "slack_user_id": {
                    "type": "string",
                    "description": "Slack user ID (optional for add)"
                },
                "slack_channel_id": {
                    "type": "string",
                    "description": "Slack channel ID (optional for add)"
                }
            },
            "required": ["action"]
        }
    },
    {
        "type": "function",
        "name": "manage_location",
        "description": "Manage location mappings - add, remove, or list locations",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "remove", "list"],
                    "description": "The action to perform"
                },
                "location": {
                    "type": "string",
                    "description": "Location name (required for add/remove)"
                },
                "videographer": {
                    "type": "string",
                    "description": "Videographer to assign the location to (required for add)"
                }
            },
            "required": ["action"]
        }
    },
    {
        "type": "function",
        "name": "manage_salesperson",
        "description": "Manage salespeople in the system - add, remove, or list salespeople",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "remove", "list"],
                    "description": "The action to perform"
                },
                "name": {
                    "type": "string",
                    "description": "Salesperson name (required for add/remove)"
                },
                "email": {
                    "type": "string",
                    "description": "Salesperson email (required for add)"
                },
                "slack_user_id": {
                    "type": "string",
                    "description": "Slack user ID (optional for add)"
                },
                "slack_channel_id": {
                    "type": "string",
                    "description": "Slack channel ID (optional for add)"
                }
            },
            "required": ["action"]
        }
    },
    {
        "type": "function",
        "name": "update_person_slack_ids",
        "description": "Update Slack user ID and/or channel ID for a person in the system. Use this when someone provides their Slack IDs after using /my_ids command.",
        "parameters": {
            "type": "object",
            "properties": {
                "person_type": {
                    "type": "string",
                    "enum": ["videographers", "sales_people", "reviewer", "hod"],
                    "description": "Type of person to update"
                },
                "person_name": {
                    "type": "string",
                    "description": "Name of the person (not required for reviewer)"
                },
                "slack_user_id": {
                    "type": "string",
                    "description": "Slack user ID (e.g., U1234567890)"
                },
                "slack_channel_id": {
                    "type": "string",
                    "description": "Slack channel ID (e.g., C1234567890)"
                }
            },
            "required": ["person_type"]
        }
    },
    {
        "type": "function",
        "name": "delete_task",
        "description": "Delete an existing task by task number. Use this when the user wants to remove, delete, or cancel an existing task. This action is permanent.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_number": {"type": "integer", "description": "The task number to delete"}
            },
            "required": ["task_number"]
        }
    },
    {
        "type": "function",
        "name": "edit_reviewer",
        "description": "Edit the reviewer's information in the system. Only super admins can use this function.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Reviewer's name"},
                "email": {"type": "string", "description": "Reviewer's email"},
                "slack_user_id": {"type": "string", "description": "Slack user ID (optional)"},
                "slack_channel_id": {"type": "string", "description": "Slack channel ID (optional)"},
                "active": {"type": "boolean", "description": "Whether the reviewer is active (optional)"}
            },
            "required": []
        }
    },
    {
        "type": "function",
        "name": "edit_hod",
        "description": "Edit the Head of Department's information in the system. Only super admins can use this function.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "HOD's name"},
                "email": {"type": "string", "description": "HOD's email"},
                "slack_user_id": {"type": "string", "description": "Slack user ID (optional)"},
                "slack_channel_id": {"type": "string", "description": "Slack channel ID (optional)"},
                "active": {"type": "boolean", "description": "Whether the HOD is active (optional)"}
            },
            "required": []
        }
    },
    {
        "type": "function",
        "name": "edit_head_of_sales",
        "description": "Edit the Head of Sales' information in the system. Only super admins can use this function.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Head of Sales' name"},
                "email": {"type": "string", "description": "Head of Sales' email"},
                "slack_user_id": {"type": "string", "description": "Slack user ID (optional)"},
                "slack_channel_id": {"type": "string", "description": "Slack channel ID (optional)"},
                "active": {"type": "boolean", "description": "Whether the Head of Sales is active (optional)"}
            },
            "required": []
        }
    }
]