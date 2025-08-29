from typing import Dict, Any
from config import UAE_TZ, SLACK_BOT_TOKEN
from clients import slack_client, client, logger
import os
import re
import json
import asyncio
from PIL import Image
from utils import _load_mapping_config, _format_sales_people_hint, _format_locations_hint, _format_videographers_hint, append_to_history, markdown_to_slack
from db_utils import save_task, check_duplicate_async as check_duplicate_reference, get_task as get_task_by_number, export_data_to_slack as export_current_data, update_task as update_task_by_number, delete_task_by_number
from history import pending_confirmations, pending_edits, pending_deletes, slash_command_responses, user_history
from management import add_videographer, remove_videographer, add_location, remove_location, list_videographers, list_locations, add_salesperson, remove_salesperson, list_salespeople, update_person_slack_ids
import requests
from datetime import datetime, timedelta
from simple_permissions import check_permission as simple_check_permission
from tools import functions
from prompts import create_edit_system_prompt, create_design_request_system_prompt

# ========== HELPER FUNCTIONS ==========
def check_admin_permission(user_id: str) -> tuple[bool, str]:
    """Check if user has admin permissions and return appropriate message"""
    # Use simple permission system
    return simple_check_permission(user_id, "manage_users")

def check_create_permission(user_id: str) -> tuple[bool, str]:
    """Check if user has create permissions and return appropriate message"""
    # Check if user can upload tasks
    return simple_check_permission(user_id, "upload_task")

def check_edit_permission(user_id: str) -> tuple[bool, str]:
    """Check if user has edit permissions and return appropriate message"""
    # Check if user can edit tasks
    return simple_check_permission(user_id, "edit_task")

# ========== TOOL FUNCTIONS ==========
async def parse_image_with_ai(file_info: Dict[str, Any], text: str = "") -> Dict[str, Any]:
    """Parse image attachment using OpenAI to extract design request information"""
    local_image_path = None
    pdf_path = None
    try:
        # Extract file details
        file_id = file_info.get("id")
        file_name = file_info.get("name", "image.png")
        
        logger.info(f"📎 Processing file: {file_name} (ID: {file_id})")
        
        # Get proper file info from Slack API
        try:
            file_response = await slack_client.files_info(file=file_id)
            if not file_response["ok"]:
                raise Exception(f"Failed to get file info: {file_response}")
            
            # Get the actual file data with proper URLs
            file_data = file_response["file"]
            file_url = file_data.get("url_private_download") or file_data.get("url_private")
            
            logger.info(f"✅ Got file info from Slack API")
            logger.debug(f"📥 Download URL: {file_url}")
            
        except Exception as e:
            logger.error(f"❌ Error getting file info from Slack: {e}")
            raise
        
        # Create uploads directory
        os.makedirs("./uploads", exist_ok=True)
        
        # Generate safe filename with timestamp
        safe_file_name = re.sub(r'[^\w\-_\.]', '_', file_name)
        timestamp = datetime.now(UAE_TZ).strftime('%Y%m%d_%H%M%S')
        local_image_path = f"./uploads/design_request_{timestamp}_{safe_file_name}"
        
        # Download using requests with Bearer token
        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        
        logger.info(f"📥 Downloading from: {file_url}")
        
        response = requests.get(file_url, headers=headers)
        response.raise_for_status()
        
        logger.info(f"📥 Downloaded {len(response.content)} bytes")
        
        # Save the image locally
        with open(local_image_path, "wb") as f_out:
            f_out.write(response.content)
        
        logger.info(f"💾 Image saved to: {local_image_path}")
        
        # Verify it's an image by checking file signature
        with open(local_image_path, 'rb') as f:
            first_bytes = f.read(20)
            if first_bytes.startswith(b'<!DOCTYPE') or first_bytes.startswith(b'<html'):
                logger.warning("⚠️ Got HTML instead of image")
                logger.debug(f"First bytes: {first_bytes}")
                raise Exception("Downloaded HTML instead of image - authentication issue")
        
        # Convert image to PDF using PIL
        def convert_image_to_pdf(input_path):
            try:
                # Open the image
                img = Image.open(input_path)
                
                # Convert to RGB if necessary (PNG might have transparency)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create a white background
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    # Paste the image on the white background
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as PDF
                pdf_path = os.path.splitext(input_path)[0] + ".pdf"
                img.save(pdf_path, 'PDF', resolution=100.0)
                
                logger.info(f"✅ Image converted to PDF successfully")
                return pdf_path
            except Exception as e:
                logger.error(f"❌ Image to PDF conversion failed: {e}")
                return None
        
        pdf_path = await asyncio.to_thread(convert_image_to_pdf, local_image_path)
        if not pdf_path:
            raise Exception("Failed to convert image to PDF")
        
        logger.info(f"📸 Image converted to PDF: {pdf_path}")
        
        # Get current date context in UAE timezone
        now = datetime.now(UAE_TZ)
        today_str = now.strftime("%B %d, %Y")
        day_of_week = now.strftime("%A")
        
        # ===== Mapping guides to improve input normalization =====
        # IMPORTANT: These are soft hints for the LLM; only map when confident.
        # Load from config dynamically

        mapping_cfg = _load_mapping_config()
        
        system_prompt = (
            "You extract design request fields from an image or PDF. "
            "Return STRICT JSON with keys: brand, campaign_start_date, campaign_end_date, reference_number, location, sales_person. "
            "Dates must be YYYY-MM-DD. Trim labels/punctuation.\n\n"
            "CRITICAL VALIDATION - YOU MUST ENFORCE:\n"
            f"1. sales_person MUST be mapped to one of: {list(mapping_cfg.get('sales_people', {}).keys())}\n"
            f"   Auto-mappings: {_format_sales_people_hint(mapping_cfg)}\n"
            f"   Common: 'Nour'→'Nourhan', 'N'→'Nourhan'\n"
            f"2. location MUST be mapped to one of: {list(mapping_cfg.get('location_mappings', {}).keys())}\n"
            f"   Valid locations: {_format_locations_hint(mapping_cfg)}\n"
            f"   Auto-map: 'TTC'→'TTC Dubai', 'Oryx'→'The Oryx', 'Gateway'→'The Gateway Dubai', '04'→'UAE 04'\n"
            "3. If you cannot map to a valid value, extract the raw text as-is (it will be validated later)\n"
            "ALWAYS attempt to map to the correct valid value when you recognize it."
        )
        
        # Upload PDF file to OpenAI (following your pattern)
        with open(pdf_path, 'rb') as file:
            file_upload = await client.files.create(
                file=file,
                purpose="user_data"
            )
        file_id = file_upload.id
        
        logger.info(f"📤 Uploaded image for parsing: {file_id}")
        
        # Build content for responses API
        content = [
            {"type": "input_file", "file_id": file_id},
            {"type": "input_text", "text": "Extract the design request information and return it as JSON."}
        ]
        
        if text:
            content.append({"type": "input_text", "text": f"Additional context: {text}"})
        
        # Use responses API with enforced JSON schema
        res = await client.responses.create(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": file_id},
                        {"type": "input_text", "text": "Extract the design request information and return it as JSON."}
                    ]
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "design_request_schema",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "brand": {
                                "type": "string",
                                "description": "The brand or client name"
                            },
                            "campaign_start_date": {
                                "type": "string",
                                "description": "The campaign start date in YYYY-MM-DD format"
                            },
                            "campaign_end_date": {
                                "type": "string",
                                "description": "The campaign end date in YYYY-MM-DD format (empty string if not found)"
                            },
                            "reference_number": {
                                "type": "string",
                                "description": "The reference number"
                            },
                            "location": {
                                "type": "string",
                                "description": "The campaign location"
                            },
                            "sales_person": {
                                "type": "string",
                                "description": "The sales person or contact name"
                            }
                        },
                        "required": ["brand", "campaign_start_date", "campaign_end_date", "reference_number", "location", "sales_person"],
                        "additionalProperties": False
                    }
                }
            },
            store=False
        )
        
        # Clean up OpenAI file only, keep local files for debugging
        await client.files.delete(file_id)
        logger.info(f"✅ Kept local files: {local_image_path} and {pdf_path}")
        
        # Parse the JSON response
        result = json.loads(res.output[0].content[-1].text)
        
        # Normalize the result
        parsed_data = {
            "brand": result.get("brand", ""),
            "start_date": result.get("campaign_start_date", ""),
            "end_date": result.get("campaign_end_date", ""),
            "reference_number": result.get("reference_number", ""),
            "location": result.get("location", "") or "",
            "sales_person": result.get("sales_person", ""),
            "timestamp": datetime.now(UAE_TZ).isoformat()
        }
        
        return parsed_data
        
    except Exception as e:
        logger.error(f"Error parsing image with AI: {e}")
        # Keep files for debugging even on error
        if local_image_path and os.path.exists(local_image_path):
            logger.warning(f"❌ Error occurred, but kept image at: {local_image_path}")
        if pdf_path and os.path.exists(pdf_path):
            logger.warning(f"❌ Error occurred, but kept PDF at: {pdf_path}")
        # Return empty structure on error
        return {
            "brand": "",
            "start_date": "",
            "end_date": "",
            "reference_number": "",
            "location": "",
            "sales_person": "",
            "timestamp": datetime.now(UAE_TZ).isoformat()
        }

def render_request_summary(data: Dict[str, Any]) -> str:
    """Render a brief Slack-friendly summary of a pending design request."""
    start_date = data.get('start_date', data.get('date', ''))
    end_date = data.get('end_date', '')
    lines = []
    lines.append(f"• Brand: `{data.get('brand', '') or 'N/A'}`")
    if start_date:
        lines.append(f"• Campaign Start Date: `{start_date}`")
    if end_date:
        lines.append(f"• Campaign End Date: `{end_date}`")
    lines.append(f"• Reference: `{data.get('reference_number', '') or 'N/A'}`")
    if data.get('location'):
        lines.append(f"• Location: `{data.get('location', '')}`")
    if data.get('sales_person'):
        lines.append(f"• Sales Person: `{data.get('sales_person', '')}`")
    return "\n".join(lines)

async def handle_design_request_confirmation(channel: str, user_id: str, user_name: str, parsed_data: Dict[str, Any], is_duplicate: bool = False) -> str:
    """Handle the entire confirmation flow with a single LLM loop until confirm/cancel"""
    # Always show the same initial format, no duplicate check here
    initial_msg = "**I've parsed the following details from your request:**\n\n"
    initial_msg += f"📋 **Brand/Client:** {parsed_data['brand']}\n"
    initial_msg += f"📅 **Campaign Start Date:** {parsed_data.get('start_date', '')}\n"
    initial_msg += f"📅 **Campaign End Date:** {parsed_data.get('end_date', '')}\n"
    initial_msg += f"🔖 **Reference:** `{parsed_data['reference_number']}`\n"
    initial_msg += f"📍 **Location:** {parsed_data['location']}\n"
    initial_msg += f"💼 **Sales Person:** {parsed_data['sales_person']}\n"
    initial_msg += f"👤 **Submitted by:** _{user_name}_\n\n"
    initial_msg += "**Is this correct?** Please confirm to save or let me know what needs to be changed."
    
    # Store in pending for the trap
    pending_confirmations[user_id] = parsed_data
    
    # Return the message to be sent by main loop
    return initial_msg

async def handle_edit_task_flow(channel: str, user_id: str, user_input: str, task_number: int, task_data: Dict[str, Any]) -> str:
    """Handle edit task flow with structured LLM response"""
    try:
        # Parse user input to determine action
        mapping_cfg = _load_mapping_config()
        system_prompt = f"""
You are helping edit Task #{task_number}. The user said: "{user_input}"

Determine their intent and parse any field updates:
- If they want to save/confirm/done: action = 'save'
- If they want to cancel/stop/exit: action = 'cancel'
- If they want to see current values: action = 'view'
- If they're making changes: action = 'edit' and parse the field updates

Current task data: {json.dumps(task_data, indent=2)}

CRITICAL VALIDATION RULES - YOU MUST ENFORCE:

1. Sales Person - ONLY accept these exact values: {list(mapping_cfg.get('sales_people', {}).keys())}
   Auto-map: {_format_sales_people_hint(mapping_cfg)}
   Common: "Nour"→"Nourhan"
   If invalid: keep current value, tell user valid options

2. Location - ONLY accept these exact values: {list(mapping_cfg.get('location_mappings', {}).keys())}
   Valid: {_format_locations_hint(mapping_cfg)}
   Auto-map: "TTC"→"TTC Dubai", "Oryx"→"The Oryx", "Gateway"→"The Gateway Dubai", "04"→"UAE 04"
   If invalid: keep current value, tell user valid options

3. Videographer - ONLY accept these exact values: {list(mapping_cfg.get('videographers', {}).keys())}
   Auto-map: "James"→"James Sevillano", "Jason"→"Jason Pieterse", "Cesar"→"Cesar Sierra", "Amr"→"Amr Tamer"
   If invalid: keep current value, tell user valid options

Return JSON with: action, fields (only changed fields with VALID values), message.
In your message, explain any fields that couldn't be updated due to invalid values.
IMPORTANT: Use natural language in messages - say 'Sales Person' not 'sales_person', 'Location' not 'location'.
"""
        
        res = await client.responses.create(
            model="gpt-4.1",
            input=[{"role": "system", "content": system_prompt}],
            text={
                'format': {
                    'type': 'json_schema',
                    'name': 'edit_response',
                    'strict': False,
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'action': {'type': 'string', 'enum': ['save', 'cancel', 'edit', 'view']},
                            'fields': {
                                'type': 'object',
                                'properties': {
                                    'Brand': {'type': 'string'},
                                    'Campaign Start Date': {'type': 'string'},
                                    'Campaign End Date': {'type': 'string'},
                                    'Reference Number': {'type': 'string'},
                                    'Location': {'type': 'string'},
                                    'Sales Person': {'type': 'string'},
                                    'Status': {'type': 'string'},
                                    'Filming Date': {'type': 'string'},
                                    'Videographer': {'type': 'string'}
                                },
                                'additionalProperties': False
                            },
                            'message': {'type': 'string'}
                        },
                        'required': ['action'],
                        'additionalProperties': False
                    }
                }
            },
            store=False
        )
        
        decision = json.loads(res.output[0].content[-1].text)
        action = decision.get('action')
        
        if action == 'save':
            # Get current updates from pending_edits
            edit_data = pending_edits.get(user_id, {})
            updates = edit_data.get('updates', {})
            
            if updates:
                # Check for duplicate reference number
                if 'Reference Number' in updates:
                    new_ref = updates['Reference Number']
                    current_ref = task_data.get('Reference Number', '')
                    
                    if new_ref != current_ref:
                        dup_check = await check_duplicate_reference(new_ref)
                        if dup_check['is_duplicate']:
                            # Show duplicate warning screen for edits
                            existing = dup_check['existing_entry']
                            warning_msg = "⚠️ **Duplicate Reference Number Detected!**\n\n"
                            warning_msg += f"The reference `{new_ref}` is already used by:\n\n"
                            warning_msg += f"• **Brand:** {existing.get('brand', existing.get('Brand', ''))}\n"
                            warning_msg += f"• **Location:** {existing.get('location', existing.get('Location', ''))}\n"
                            warning_msg += f"• **Campaign:** {existing.get('start_date', existing.get('Campaign Start Date', ''))} to {existing.get('end_date', existing.get('Campaign End Date', ''))}\n\n"
                            warning_msg += "**Do you want to proceed with this duplicate?**\n"
                            warning_msg += "• Say **'yes'** or **'save anyway'** to update with duplicate\n"
                            warning_msg += "• Say **'no'** or **'cancel'** to cancel the edit\n"
                            warning_msg += "• Say **'edit'** to continue editing and change the reference"
                            
                            # Mark that we're in duplicate confirmation mode for edits
                            edit_data['_duplicate_confirm'] = True
                            pending_edits[user_id] = edit_data
                            return warning_msg
                
                result = await update_task_by_number(task_number, updates, task_data)
                if result["success"]:
                    del pending_edits[user_id]
                    answer = f"✅ **Task #{task_number} updated successfully!**\n\n**Changes made:**\n"
                    for field, value in updates.items():
                        answer += f"• {field}: {task_data.get(field, 'N/A')} → {value}\n"
                    if "warning" in result:
                        answer += f"\n⚠️ Note: {result['warning']}"
                    return answer
                else:
                    return f"❌ Error updating Task #{task_number}: {result.get('error', 'Unknown error')}"
            else:
                del pending_edits[user_id]
                return f"No changes were made to Task #{task_number}."
        
        elif action == 'cancel':
            del pending_edits[user_id]
            return f"❌ Edit cancelled for Task #{task_number}. No changes were saved."
        
        elif action == 'edit':
            # Apply the edits
            fields = decision.get('fields', {})
            if fields:
                # Get or create edit data
                if user_id not in pending_edits:
                    pending_edits[user_id] = {
                        "task_number": task_number,
                        "current_data": task_data,
                        "updates": {}
                    }
                
                # Apply only changed fields
                actual_updates = {}
                for field, new_value in fields.items():
                    if new_value and str(task_data.get(field, '')) != str(new_value):
                        # Format dates if needed
                        if field in ['Campaign Start Date', 'Campaign End Date', 'Filming Date'] and new_value:
                            if len(new_value) == 10 and new_value[4] == '-':
                                try:
                                    date_obj = datetime.strptime(new_value, "%Y-%m-%d")
                                    new_value = date_obj.strftime("%d-%m-%Y")
                                except:
                                    pass
                        actual_updates[field] = new_value
                
                if actual_updates:
                    pending_edits[user_id]["updates"].update(actual_updates)
                    
                    answer = "✅ **Updates recorded:**\n"
                    for field, value in actual_updates.items():
                        answer += f"• {field}: {value}\n"
                    answer += f"\n**Total pending changes for Task #{task_number}:**\n"
                    for field, value in pending_edits[user_id]["updates"].items():
                        answer += f"• {field}: {task_data.get(field, 'N/A')} → {value}\n"
                    answer += "\nContinue editing or say 'save' when done."
                    return answer
            
            return decision.get('message', 'Please provide the changes you want to make.')
        
        elif action == 'view':
            # Show current data with any pending updates
            edit_data = pending_edits.get(user_id, {})
            updates = edit_data.get('updates', {})
            
            answer = f"**Current data for Task #{task_number}:**\n"
            for field in ['Brand', 'Campaign Start Date', 'Campaign End Date', 'Reference Number', 'Location', 'Sales Person', 'Status', 'Filming Date', 'Videographer']:
                current = task_data.get(field, 'N/A')
                if field in updates:
                    answer += f"• {field}: {current} → **{updates[field]}** (pending)\n"
                else:
                    answer += f"• {field}: {current}\n"
            answer += "\nContinue editing or say 'save' when done."
            return answer
        
        else:
            return "Please provide changes, say 'save' to commit, or 'cancel' to exit."
            
    except Exception as e:
        logger.error(f"Error in edit task flow: {e}")
        return "❌ Error processing your edit. Please try again."

async def handle_confirmation_response(channel: str, user_id: str, user_input: str, pending_data: Dict[str, Any]) -> str:
    """Handle user response during confirmation using structured LLM response"""
    try:
        today_str = datetime.now(UAE_TZ).strftime('%B %d, %Y')
        
        # Check if we're in duplicate confirmation mode
        if pending_data.get('_duplicate_confirm'):
            # Use LLM to parse duplicate confirmation response
            dup_system_prompt = (
                "The user is responding to a duplicate reference number warning. "
                "They were told the reference already exists and asked if they want to proceed. "
                "Parse their response and determine their intent:\n"
                "- If they want to proceed/accept/yes/confirm the duplicate: action = 'accept'\n"
                "- If they want to cancel/no/stop: action = 'cancel'\n"
                "- If they want to edit/change/modify the reference: action = 'edit'\n"
                "Return JSON with: action (accept/cancel/edit), message"
            )
            
            try:
                res = await client.responses.create(
                    model="gpt-4.1",
                    input=[
                        {"role": "system", "content": dup_system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    text={
                        'format': {
                            'type': 'json_schema',
                            'name': 'duplicate_response',
                            'strict': False,
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'action': {'type': 'string', 'enum': ['accept', 'cancel', 'edit']},
                                    'message': {'type': 'string'}
                                },
                                'required': ['action'],
                                'additionalProperties': False
                            }
                        }
                    },
                    store=False
                )
                
                dup_decision = json.loads(res.output[0].content[-1].text)
                dup_action = dup_decision.get('action')
                
                if dup_action == 'accept':
                    # Remove the flag and save
                    del pending_data['_duplicate_confirm']
                    result = await save_task(pending_data)
                    if result["success"]:
                        del pending_confirmations[user_id]
                        task_number = result["task_number"]
                        brand = pending_data.get('brand', '')
                        reference = pending_data.get('reference_number', '')
                        return f"✅ **Task #{task_number} created successfully!**\n\n**{brand} - Design Request**\n• Reference: `{reference}` _(duplicate accepted)_\n• Campaign: {pending_data.get('start_date', '')} to {pending_data.get('end_date', '')}\n• Location: {pending_data.get('location', '')}\n• Sales Person: {pending_data.get('sales_person', '')}"
                    else:
                        return "❌ Error saving your request. Please try again."
                elif dup_action == 'cancel':
                    del pending_confirmations[user_id]
                    return dup_decision.get('message', "❌ Request cancelled due to duplicate reference number.")
                elif dup_action == 'edit':
                    # Go back to edit mode
                    del pending_data['_duplicate_confirm']
                    pending_confirmations[user_id] = pending_data
                    formatted_msg = "**Current details:**\n\n"
                    formatted_msg += f"📋 **Brand/Client:** {pending_data.get('brand', '')}\n"
                    formatted_msg += f"📅 **Campaign Start Date:** {pending_data.get('start_date', '')}\n"
                    formatted_msg += f"📅 **Campaign End Date:** {pending_data.get('end_date', '')}\n"
                    formatted_msg += f"🔖 **Reference:** `{pending_data.get('reference_number', '')}` ⚠️ _duplicate_\n"
                    formatted_msg += f"📍 **Location:** {pending_data.get('location', '')}\n"
                    formatted_msg += f"💼 **Sales Person:** {pending_data.get('sales_person', '')}\n\n"
                    formatted_msg += "Please provide your changes, especially the reference number."
                    return formatted_msg
                
            except Exception as e:
                logger.error(f"Error parsing duplicate confirmation: {e}")
                return "I didn't understand your response. Please say 'yes' to proceed with duplicate, 'no' to cancel, or 'edit' to change."
        
        # Build conversation context
        current_data_text = render_request_summary(pending_data)
        
        mapping_cfg = _load_mapping_config()
        system_prompt = (
            "You are helping confirm or edit a design request. "
            f"Today's date is {today_str}. "
            "Analyze the user's response and determine the action. "
            "Return JSON with: action (confirm/cancel/edit/view), fields (only for edits), message. "
            "Actions: "
            "- confirm: User agrees (yes, yup, confirm, proceed, correct, looks good, etc.) "
            "- cancel: User cancels (no, cancel, stop, nevermind, etc.) "
            "- edit: User provides corrections (include ONLY changed fields) "
            "- view: User wants to see current details (show, view, display, what are the details, etc.) "
            "For edit action, parse the corrections and include in 'fields'. "
            "\nCRITICAL VALIDATION FOR EDITS:\n"
            f"- sales_person MUST be one of: {list(mapping_cfg.get('sales_people', {}).keys())}\n"
            f"  Auto-map: {_format_sales_people_hint(mapping_cfg)}, 'Nour'→'Nourhan'\n"
            f"- location MUST be one of: {list(mapping_cfg.get('location_mappings', {}).keys())}\n"
            f"  Auto-map: 'TTC'→'TTC Dubai', 'Oryx'→'The Oryx', 'Gateway'→'The Gateway Dubai', '04'→'UAE 04'\n"
            "If user provides invalid values after mapping, do NOT include them in fields.\n"
            "In your message, explain which fields couldn't be updated and list valid options.\n"
            "IMPORTANT: Use natural language in your messages - say 'Sales Person' not 'sales_person', 'Location' not 'location'.\n"
            "Always provide a helpful 'message' to guide the user."
        )
        
        # Use structured output
        res = await client.responses.create(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Current design request:\n{current_data_text}\n\nUser response: {user_input}"}
            ],
            text={
                'format': {
                    'type': 'json_schema',
                    'name': 'confirmation_response',
                    'strict': False,
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'action': {'type': 'string', 'enum': ['confirm', 'cancel', 'edit', 'view']},
                            'fields': {
                                'type': 'object',
                                'properties': {
                                    'brand': {'type': 'string'},
                                    'start_date': {'type': 'string'},
                                    'end_date': {'type': 'string'},
                                    'reference_number': {'type': 'string'},
                                    'location': {'type': 'string'},
                                    'sales_person': {'type': 'string'}
                                },
                                'additionalProperties': False
                            },
                            'message': {'type': 'string'}
                        },
                        'required': ['action', 'message'],
                        'additionalProperties': False
                    }
                }
            },
            store=False
        )
        
        decision = json.loads(res.output[0].content[-1].text)
        action = decision.get('action')
        message = decision.get('message', '')

        if action == 'confirm':
            # Check for duplicate only when confirming
            dup_check = await check_duplicate_reference(pending_data['reference_number'])
            
            if dup_check['is_duplicate']:
                # Show duplicate warning screen
                existing = dup_check['existing_entry']
                warning_msg = "⚠️ **Duplicate Reference Number Detected!**\n\n"
                warning_msg += f"The reference `{pending_data['reference_number']}` is already used by:\n\n"
                warning_msg += f"• **Brand:** {existing.get('brand', existing.get('Brand', ''))}\n"
                warning_msg += f"• **Location:** {existing.get('location', existing.get('Location', ''))}\n"
                warning_msg += f"• **Campaign:** {existing.get('start_date', existing.get('Campaign Start Date', ''))} to {existing.get('end_date', existing.get('Campaign End Date', ''))}\n\n"
                warning_msg += "**Do you want to proceed with this duplicate?**\n"
                warning_msg += "• Say **'yes'** or **'accept'** to create anyway\n"
                warning_msg += "• Say **'no'** or **'cancel'** to cancel\n"
                warning_msg += "• Say **'edit'** to go back and change the reference"
                
                # Mark that we're in duplicate confirmation mode
                pending_data['_duplicate_confirm'] = True
                pending_confirmations[user_id] = pending_data
                return warning_msg
            
            # Save and exit (no duplicate)
            result = await save_task(pending_data)
            if result["success"]:
                if user_id in pending_confirmations:
                    del pending_confirmations[user_id]
                task_number = result["task_number"]
                brand = pending_data.get('brand', '')
                reference = pending_data.get('reference_number', '')
                return f"✅ **Task #{task_number} created successfully!**\n\n**{brand} - Design Request**\n• Reference: `{reference}`\n• Campaign: {pending_data.get('start_date', '')} to {pending_data.get('end_date', '')}\n• Location: {pending_data.get('location', '')}\n• Sales Person: {pending_data.get('sales_person', '')}"
            else:
                return "❌ Error saving your request. Please try again."
        
        elif action == 'cancel':
            if user_id in pending_confirmations:
                del pending_confirmations[user_id]
            return message or "❌ Request cancelled. No data was saved."
        
        elif action == 'edit':
            # Apply edits
            fields = decision.get('fields', {})
            if fields:
                for key, value in fields.items():
                    if value and value.strip():
                        pending_data[key] = value.strip()
            
            # Format the updated data in the same style as initial confirmation
            formatted_msg = "\n\n**Updated details:**\n\n"
            formatted_msg += f"📋 **Brand/Client:** {pending_data.get('brand', '')}\n"
            formatted_msg += f"📅 **Campaign Start Date:** {pending_data.get('start_date', '')}\n"
            formatted_msg += f"📅 **Campaign End Date:** {pending_data.get('end_date', '')}\n"
            formatted_msg += f"🔖 **Reference:** `{pending_data.get('reference_number', '')}`\n"
            formatted_msg += f"📍 **Location:** {pending_data.get('location', '')}\n"
            formatted_msg += f"💼 **Sales Person:** {pending_data.get('sales_person', '')}\n"
            
            return f"{message}{formatted_msg}\nPlease confirm or continue editing."
        
        elif action == 'view':
            # Show current details in the same style
            formatted_msg = "**Current details:**\n\n"
            formatted_msg += f"📋 **Brand/Client:** {pending_data.get('brand', '')}\n"
            formatted_msg += f"📅 **Campaign Start Date:** {pending_data.get('start_date', '')}\n"
            formatted_msg += f"📅 **Campaign End Date:** {pending_data.get('end_date', '')}\n"
            formatted_msg += f"🔖 **Reference:** `{pending_data.get('reference_number', '')}`\n"
            formatted_msg += f"📍 **Location:** {pending_data.get('location', '')}\n"
            formatted_msg += f"💼 **Sales Person:** {pending_data.get('sales_person', '')}\n"
            
            return f"{formatted_msg}\nPlease confirm if correct or let me know what to change."
        
        else:
            return message or "Please say 'confirm' to save, provide corrections, or 'cancel' to stop."
        
    except Exception as e:
        logger.error(f"Error in confirmation handler: {e}")
        return "❌ Error processing your response. Please try again."


# ========== MAIN LLM LOOP ==========
async def main_llm_loop(channel: str, user_id: str, user_input: str, files: list = None):
    """Main conversational loop with OpenAI responses API"""
    # Get user's display name
    try:
        res = await slack_client.users_info(user=user_id)
        profile = res["user"]["profile"]
        name = profile.get("display_name") or profile.get("real_name") or "there"
    except:
        name = "there"
    
    now = datetime.now(UAE_TZ)
    today_str = now.strftime("%B %d, %Y")
    day_of_week = now.strftime("%A")
    
    # Check if user has a pending confirmation (handles both normal and duplicate)
    if user_id in pending_confirmations:
        # Use the new confirmation handler
        answer = await handle_confirmation_response(channel, user_id, user_input, pending_confirmations[user_id])
        
        # Check if this is a slash command response
        if user_id in slash_command_responses:
            response_url = slash_command_responses[user_id]
            if user_id not in pending_confirmations:  # Only clean up if confirmation is done
                del slash_command_responses[user_id]
            
            # Send response using the slash command response URL
            response_data = {
                "response_type": "in_channel",
                "text": markdown_to_slack(answer)
            }
            
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    response_url,
                    json=response_data
                )
                response.raise_for_status()
                logger.info(f"✅ Sent slash command confirmation response to {response_url}")
            except Exception as e:
                logger.error(f"❌ Failed to send slash command confirmation response: {e}")
                await slack_client.chat_postMessage(channel=channel, text=markdown_to_slack(answer))
        else:
            await slack_client.chat_postMessage(channel=channel, text=markdown_to_slack(answer))
        
        append_to_history(user_id, "user", user_input)
        append_to_history(user_id, "assistant", answer)
        return

    # Check if user has a pending delete
    elif user_id in pending_deletes:
        # Handle delete confirmation
        delete_data = pending_deletes[user_id]
        task_number = delete_data["task_number"]
        task_data = delete_data["task_data"]
        
        # Use LLM to parse delete confirmation response
        delete_prompt = (
            f"The user is confirming whether to delete Task #{task_number}. "
            "Parse their response and determine their intent:\n"
            "- If they want to delete/confirm/yes/proceed: action = 'confirm'\n"
            "- If they want to cancel/no/stop/keep: action = 'cancel'\n"
            "Return JSON with: action (confirm/cancel), message"
        )
        
        try:
            res = await client.responses.create(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": delete_prompt},
                    {"role": "user", "content": user_input}
                ],
                text={
                    'format': {
                        'type': 'json_schema',
                        'name': 'delete_confirmation',
                        'strict': False,
                        'schema': {
                            'type': 'object',
                            'properties': {
                                'action': {'type': 'string', 'enum': ['confirm', 'cancel']},
                                'message': {'type': 'string'}
                            },
                            'required': ['action'],
                            'additionalProperties': False
                        }
                    }
                },
                store=False
            )
            
            decision = json.loads(res.output[0].content[-1].text)
            action = decision.get('action')
            
            if action == 'confirm':
                # Delete the task
                result = await delete_task_by_number(task_number)
                
                if result["success"]:
                    del pending_deletes[user_id]
                    deleted_data = result["task_data"]
                    answer = f"✅ **Task #{task_number} has been deleted successfully!**\n\n"
                    answer += "**Deleted task details:**\n"
                    answer += f"• Brand: {deleted_data.get('Brand', 'N/A')}\n"
                    answer += f"• Reference: {deleted_data.get('Reference Number', 'N/A')}\n"
                    answer += f"• Location: {deleted_data.get('Location', 'N/A')}\n"
                    answer += f"• Campaign: {deleted_data.get('Campaign Start Date', 'N/A')} to {deleted_data.get('Campaign End Date', 'N/A')}\n"
                    answer += f"• Status: {deleted_data.get('Status', 'N/A')}\n"
                    
                    if str(deleted_data.get('Status', '')).startswith('Assigned to'):
                        answer += "\n✅ The associated Trello card has been archived."
                    
                    answer += "\n\n_The task has been archived in the history database._"
                else:
                    answer = f"❌ Error deleting Task #{task_number}: {result.get('error', 'Unknown error')}"
            else:
                # Cancel deletion
                del pending_deletes[user_id]
                answer = f"❌ Deletion cancelled. Task #{task_number} has been kept."
                
        except Exception as e:
            logger.error(f"Error parsing delete confirmation: {e}")
            answer = "I didn't understand your response. Please say 'yes' to delete or 'no' to cancel."
        
        await slack_client.chat_postMessage(channel=channel, text=markdown_to_slack(answer))
        append_to_history(user_id, "user", user_input)
        append_to_history(user_id, "assistant", answer)
        return

    # Check if user has a pending edit
    elif user_id in pending_edits:
        # Handle edit mode using the new flow
        edit_data = pending_edits[user_id]
        task_number = edit_data["task_number"]
        current_data = edit_data["current_data"]
        
        # Check if we're in duplicate confirmation mode for edits
        if edit_data.get('_duplicate_confirm'):
            # Use LLM to parse duplicate confirmation response for edits
            dup_edit_prompt = (
                f"The user is editing Task #{task_number} and tried to change the reference number to a duplicate. "
                "They were warned about the duplicate and asked how to proceed. "
                "Parse their response and determine their intent:\n"
                "- If they want to proceed/save anyway/accept the duplicate: action = 'accept'\n"
                "- If they want to cancel the edit: action = 'cancel'\n"
                "- If they want to continue editing/change the reference: action = 'edit'\n"
                "Return JSON with: action (accept/cancel/edit), message"
            )
            try:
                res = await client.responses.create(
                    model="gpt-4.1",
                    input=[
                        {"role": "system", "content": dup_edit_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    text={
                        'format': {
                            'type': 'json_schema',
                            'name': 'duplicate_edit_response',
                            'strict': False,
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'action': {'type': 'string', 'enum': ['accept', 'cancel', 'edit']},
                                    'message': {'type': 'string'}
                                },
                                'required': ['action'],
                                'additionalProperties': False
                            }
                        }
                    },
                    store=False
                )
                dup_decision = json.loads(res.output[0].content[-1].text)
                dup_action = dup_decision.get('action')
                if dup_action == 'accept':
                    # Remove flag and save with duplicate
                    del edit_data['_duplicate_confirm']
                    updates = edit_data.get('updates', {})
                    result = await update_task_by_number(task_number, updates, current_data)
                    if result["success"]:
                        del pending_edits[user_id]
                        answer = f"✅ **Task #{task_number} updated successfully!**\n\n**Changes made:**\n"
                        for field, value in updates.items():
                            answer += f"• {field}: {current_data.get(field, 'N/A')} → {value}\n"
                        answer += f"\n⚠️ _Note: Reference number is a duplicate_"
                    else:
                        answer = f"❌ Error updating Task #{task_number}: {result.get('error', 'Unknown error')}"
                elif dup_action == 'cancel':
                    del pending_edits[user_id]
                    answer = dup_decision.get('message', f"❌ Edit cancelled for Task #{task_number}.")
                elif dup_action == 'edit':
                    # Go back to edit mode
                    del edit_data['_duplicate_confirm']
                    pending_edits[user_id] = edit_data
                    answer = f"**Current pending changes:**\n"
                    for field, value in edit_data.get('updates', {}).items():
                        answer += f"• {field}: {current_data.get(field, 'N/A')} → {value}\n"
                    answer += "\nPlease provide your changes, especially a new reference number."
                else:
                    answer = "I didn't understand. Please say 'save' to proceed with duplicate, 'cancel' to stop, or 'edit' to continue."
            except Exception as e:
                logger.error(f"Error parsing edit duplicate confirmation: {e}")
                answer = "I didn't understand your response. Please say 'save' to proceed, 'cancel' to stop, or 'edit' to continue."
        else:
            answer = await handle_edit_task_flow(channel, user_id, user_input, task_number, current_data)
        
        await slack_client.chat_postMessage(channel=channel, text=markdown_to_slack(answer))
        append_to_history(user_id, "user", user_input)
        append_to_history(user_id, "assistant", answer)
        return

    system_prompt = create_design_request_system_prompt(name)

    history = user_history.get(user_id, [])
    
    # Handle image files if provided
    user_content = user_input
    if files:
        # Process image files for OpenAI
        for file in files:
            if file.get("mimetype", "").startswith("image/"):
                # Download and convert image to PDF for OpenAI
                try:
                    file_id = file.get("id")
                    file_name = file.get("name", "image.png")
                    
                    # Get file info from Slack
                    file_response = await slack_client.files_info(file=file_id)
                    if file_response["ok"]:
                        file_data = file_response["file"]
                        file_url = file_data.get("url_private_download") or file_data.get("url_private")
                        
                        # Download the file
                        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
                        response = requests.get(file_url, headers=headers)
                        response.raise_for_status()
                        
                        # Save temporarily
                        os.makedirs("./uploads", exist_ok=True)
                        timestamp = datetime.now(UAE_TZ).strftime('%Y%m%d_%H%M%S')
                        safe_file_name = re.sub(r'[^\w\-_\.]', '_', file_name)
                        local_path = f"./uploads/temp_{timestamp}_{safe_file_name}"
                        
                        with open(local_path, "wb") as f:
                            f.write(response.content)
                        
                        # Convert to PDF if needed
                        if not file_name.lower().endswith('.pdf'):
                            # Convert image to PDF
                            img = Image.open(local_path)
                            if img.mode in ('RGBA', 'LA', 'P'):
                                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                                if img.mode == 'P':
                                    img = img.convert('RGBA')
                                rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                                img = rgb_img
                            elif img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            pdf_path = os.path.splitext(local_path)[0] + ".pdf"
                            img.save(pdf_path, 'PDF', resolution=100.0)
                            os.remove(local_path)  # Remove original
                            local_path = pdf_path
                        
                        # Upload to OpenAI
                        with open(local_path, 'rb') as f:
                            file_upload = await client.files.create(
                                file=f,
                                purpose="user_data"
                            )
                        
                        # Clean up local file
                        os.remove(local_path)
                        
                        # Update user content to include file
                        user_content = [
                            {"type": "input_text", "text": user_input},
                            {"type": "input_file", "file_id": file_upload.id}
                        ]
                        
                        # Schedule cleanup after a delay to ensure OpenAI has processed it
                        async def cleanup_file():
                            await asyncio.sleep(60)  # Wait 60 seconds
                            try:
                                await client.files.delete(file_upload.id)
                                logger.info(f"Cleaned up OpenAI file {file_upload.id}")
                            except Exception as e:
                                logger.error(f"Error cleaning up file: {e}")
                        
                        asyncio.create_task(cleanup_file())
                        
                except Exception as e:
                    logger.error(f"Error processing image for LLM: {e}")
                    user_content = user_input + f"\n\n[Note: Image upload failed: {str(e)}]"
    
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_content}]

    # Print input to OpenAI (for debug)
    logger.debug("\n🧠 --- Prompt Sent to OpenAI ---")
    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        # Handle both string and list content
        if isinstance(msg["content"], str):
            content = msg["content"].strip().replace("\n", " ")
        elif isinstance(msg["content"], list):
            # For multimodal content, show text part and note about file
            text_parts = [part["text"] for part in msg["content"] if part.get("type") == "input_text"]
            file_parts = [part for part in msg["content"] if part.get("type") == "input_file"]
            content = " ".join(text_parts) + (f" [+{len(file_parts)} file(s)]" if file_parts else "")
        else:
            content = str(msg["content"])
        icon = {"USER": "🧑", "ASSISTANT": "🤖", "SYSTEM": "⚙️"}.get(role, "❓")
        logger.debug(f"{i+1:02d}. {icon} [{role}]: {content[:200] if len(content) > 200 else content}")
    logger.debug("🔚 --- End of Prompt ---\n")

    try:
        # Call OpenAI with responses API
        res = await client.responses.create(
            model="gpt-4.1",
            input=messages,
            tools=functions,
            tool_choice="auto"
        )

        msg = res.output[0]
        
        if msg.type == "function_call":
            # Handle function calls
            func_name = msg.name
            args = json.loads(msg.arguments)
            logger.debug(f"\n🛠️ Tool Called: {func_name} with args: {args}")

            if func_name == "log_design_request":
                # Check permissions
                has_permission, error_msg = check_create_permission(user_id)
                if not has_permission:
                    answer = error_msg
                else:
                    # Store the parsed data for confirmation
                    parsed_data = {
                        "brand": args["brand"],
                        "start_date": args.get("campaign_start_date") or args.get("campaign_date") or "",
                        "end_date": args.get("campaign_end_date", ""),
                        "reference_number": args["reference_number"],
                        "location": args.get("location", ""),
                        "sales_person": args.get("sales_person", ""),
                        "submitted_by": name
                    }
                    
                    # Format dates from YYYY-MM-DD to DD-MM-YYYY if needed
                    for date_field in ['start_date', 'end_date']:
                        if parsed_data[date_field] and len(parsed_data[date_field]) == 10 and parsed_data[date_field][4] == '-':
                            try:
                                date_obj = datetime.strptime(parsed_data[date_field], "%Y-%m-%d")
                                parsed_data[date_field] = date_obj.strftime("%d-%m-%Y")
                            except:
                                pass
                    
                    # Use the new confirmation handler (no duplicate check here)
                    answer = await handle_design_request_confirmation(
                        channel=channel,
                        user_id=user_id,
                        user_name=name,
                        parsed_data=parsed_data,
                        is_duplicate=False  # Don't check for duplicates until confirm
                    )
                
            elif func_name == "export_current_data":
                # Check permissions for viewing excel/db
                has_permission, error_msg = simple_check_permission(user_id, "view_excel")
                if not has_permission:
                    answer = error_msg
                else:
                    answer = await export_current_data(
                        include_history=args.get("include_history", True),
                        channel=channel,
                        user_id=user_id
                    )
            elif func_name == "edit_task":
                # Check permissions
                has_permission, error_msg = check_edit_permission(user_id)
                if not has_permission:
                    answer = error_msg
                else:
                    task_number = args.get("task_number")
                    
                    # Get the task data
                    task_data = await get_task_by_number(task_number)
                    
                    if task_data:
                        # Enter edit mode
                        pending_edits[user_id] = {
                            "task_number": task_number,
                            "current_data": task_data,
                            "updates": {}
                        }
                        
                        answer = f"📝 **Editing Task #{task_number}**\n\n"
                        answer += "**Current task data:**\n"
                        answer += f"• Brand: {task_data.get('Brand', 'N/A')}\n"
                        answer += f"• Campaign Start Date: {task_data.get('Campaign Start Date', 'N/A')}\n"
                        answer += f"• Campaign End Date: {task_data.get('Campaign End Date', 'N/A')}\n"
                        answer += f"• Reference Number: {task_data.get('Reference Number', 'N/A')}\n"
                        answer += f"• Location: {task_data.get('Location', 'N/A')}\n"
                        answer += f"• Sales Person: {task_data.get('Sales Person', 'N/A')}\n"
                        answer += f"• Status: {task_data.get('Status', 'N/A')}\n"
                        answer += f"• Filming Date: {task_data.get('Filming Date', 'N/A')}\n"
                        if task_data.get('Videographer'):
                            answer += f"• Videographer: {task_data.get('Videographer', 'N/A')}\n"
                        if task_data.get('Video Filename'):
                            answer += f"• Video Filename: {task_data.get('Video Filename', 'N/A')}\n"
                        answer += "\nTell me what you'd like to change. You can update multiple fields, and let me know when you're done."
                    else:
                        answer = f"I couldn't find Task #{task_number}. Please check the task number and try again, or use /recent_requests to see available tasks."
            
            elif func_name == "manage_videographer":
                # Check permissions for videographer management
                action = args.get("action", "list")
                if action == "list":
                    has_permission, error_msg = simple_check_permission(user_id, "view_excel")
                elif action == "add":
                    has_permission, error_msg = simple_check_permission(user_id, "manage_users")
                elif action == "remove":
                    has_permission, error_msg = simple_check_permission(user_id, "manage_users")
                else:
                    has_permission, error_msg = False, "Unknown action"
                
                if not has_permission:
                    answer = error_msg
                else:
                    action = args.get("action")
                    
                    if action == "add":
                        name = args.get("name")
                        email = args.get("email")
                        slack_user_id = args.get("slack_user_id", "")
                        slack_channel_id = args.get("slack_channel_id", "")
                        
                        if not name or not email:
                            answer = "Please provide both name and email for the new videographer."
                        else:
                            result = await add_videographer(name, email, slack_user_id, slack_channel_id)
                            if result["success"]:
                                answer = f"✅ {result['message']}\n\nVideographer '{name}' has been added:\n"
                                answer += f"• Email: {email}\n"
                                if slack_user_id:
                                    answer += f"• Slack User ID: {slack_user_id}\n"
                                if slack_channel_id:
                                    answer += f"• Slack Channel ID: {slack_channel_id}\n"
                                answer += "\nA Trello list has been created for them."
                            else:
                                answer = f"❌ Failed to add videographer: {result['error']}"
                    
                    elif action == "remove":
                        name = args.get("name")
                        if not name:
                            answer = "Please provide the name of the videographer to remove."
                        else:
                            result = await remove_videographer(name)
                            if result["success"]:
                                answer = f"✅ {result['message']}\n\nVideographer '{name}' has been removed and their Trello list archived."
                            else:
                                answer = f"❌ Failed to remove videographer: {result['error']}"
                                if "assigned_locations" in result:
                                    answer += f"\n\nAssigned locations: {', '.join(result['assigned_locations'])}"
                                if "active_tasks" in result:
                                    answer += f"\n\nActive tasks: {result['active_tasks']}"
                    
                    elif action == "list":
                        result = await list_videographers()
                        if result["success"]:
                            answer = f"📋 **Videographer List** ({result['total_videographers']} total)\n\n"
                            for name, details in result["videographers"].items():
                                count = result["location_counts"].get(name, 0)
                                answer += f"• **{name}**\n"
                                answer += f"  - Email: {details['email']}\n"
                                answer += f"  - Status: {'Active' if details.get('active', True) else 'Inactive'}\n"
                                answer += f"  - Assigned Locations: {count}\n\n"
                        else:
                            answer = f"❌ Error listing videographers: {result['error']}"
            
            elif func_name == "manage_location":
                # Check permissions for location management
                action = args.get("action", "list")
                if action == "list":
                    has_permission, error_msg = simple_check_permission(user_id, "view_excel")
                elif action == "add":
                    has_permission, error_msg = simple_check_permission(user_id, "manage_users")
                elif action == "remove":
                    has_permission, error_msg = simple_check_permission(user_id, "manage_users")
                else:
                    has_permission, error_msg = False, "Unknown action"
                
                if not has_permission:
                    answer = error_msg
                else:
                    action = args.get("action")
                    
                    if action == "add":
                        location = args.get("location")
                        videographer = args.get("videographer")
                        if not location or not videographer:
                            answer = "Please provide both location name and videographer to assign it to."
                        else:
                            result = await add_location(location, videographer)
                            if result["success"]:
                                answer = f"✅ {result['message']}\n\nLocation '{location}' is now assigned to {videographer}."
                            else:
                                answer = f"❌ Failed to add location: {result['error']}"
                    
                    elif action == "remove":
                        location = args.get("location")
                        if not location:
                            answer = "Please provide the location name to remove."
                        else:
                            result = await remove_location(location)
                            if result["success"]:
                                answer = f"✅ {result['message']}"
                            else:
                                answer = f"❌ Failed to remove location: {result['error']}"
                                if "unassigned_tasks" in result:
                                    answer += f"\n\nUnassigned tasks at this location: {result['unassigned_tasks']}"
                    
                    elif action == "list":
                        result = await list_locations()
                        if result["success"]:
                            answer = f"📍 **Location Mappings** ({result['total_locations']} total)\n\n"
                            # Group locations by videographer
                            by_videographer = {}
                            for loc, vid in result["location_mappings"].items():
                                if vid not in by_videographer:
                                    by_videographer[vid] = []
                                by_videographer[vid].append(loc)
                            
                            for videographer, locations in sorted(by_videographer.items()):
                                answer += f"**{videographer}:**\n"
                                for loc in sorted(locations):
                                    answer += f"  • {loc}\n"
                                answer += "\n"
                        else:
                            answer = f"❌ Error listing locations: {result['error']}"
            
            elif func_name == "manage_salesperson":
                # Check permissions for salesperson management
                action = args.get("action", "list")
                if action == "list":
                    has_permission, error_msg = simple_check_permission(user_id, "view_excel")
                elif action == "add":
                    has_permission, error_msg = simple_check_permission(user_id, "manage_users")
                elif action == "remove":
                    has_permission, error_msg = simple_check_permission(user_id, "manage_users")
                else:
                    has_permission, error_msg = False, "Unknown action"
                
                if not has_permission:
                    answer = error_msg
                else:
                    action = args.get("action")
                    
                    if action == "add":
                        name = args.get("name")
                        email = args.get("email")
                        slack_user_id = args.get("slack_user_id", "")
                        slack_channel_id = args.get("slack_channel_id", "")
                        
                        if not name or not email:
                            answer = "Please provide both name and email for the new salesperson."
                        else:
                            result = await add_salesperson(name, email, slack_user_id, slack_channel_id)
                            if result["success"]:
                                answer = f"✅ {result['message']}\n\nSalesperson '{name}' has been added:\n"
                                answer += f"• Email: {email}\n"
                                if slack_user_id:
                                    answer += f"• Slack User ID: {slack_user_id}\n"
                                if slack_channel_id:
                                    answer += f"• Slack Channel ID: {slack_channel_id}\n"
                            else:
                                answer = f"❌ Failed to add salesperson: {result['error']}"
                    
                    elif action == "remove":
                        name = args.get("name")
                        if not name:
                            answer = "Please provide the name of the salesperson to remove."
                        else:
                            result = await remove_salesperson(name)
                            if result["success"]:
                                answer = f"✅ {result['message']}"
                            else:
                                answer = f"❌ Failed to remove salesperson: {result['error']}"
                                if "active_tasks" in result:
                                    answer += f"\n\nActive tasks for this salesperson: {result['active_tasks']}"
                    
                    elif action == "list":
                        result = await list_salespeople()
                        if result["success"]:
                            answer = f"💼 **Salespeople** ({result['total_salespeople']} total)\n\n"
                            for salesperson, info in sorted(result["salespeople"].items()):
                                answer += f"**{salesperson}**\n"
                                answer += f"  • Email: {info['email']}\n"
                                answer += f"  • Status: {'Active' if info.get('active', True) else 'Inactive'}\n"
                                if salesperson in result.get("task_counts", {}):
                                    answer += f"  • Active Tasks: {result['task_counts'][salesperson]}\n"
                                answer += "\n"
                        else:
                            answer = f"❌ Error listing salespeople: {result['error']}"
            
            elif func_name == "update_person_slack_ids":
                # Check permissions for updating Slack IDs
                has_permission, error_msg = simple_check_permission(user_id, "update_slack_ids")
                if not has_permission:
                    answer = error_msg
                else:
                    person_type = args.get("person_type")
                    person_name = args.get("person_name", "")
                    slack_user_id = args.get("slack_user_id")
                    slack_channel_id = args.get("slack_channel_id")
                    
                    if not slack_user_id and not slack_channel_id:
                        answer = "Please provide at least one Slack ID to update (user ID or channel ID)."
                    elif person_type not in ["reviewer", "hod"] and not person_name:
                        answer = f"Please provide the name of the {person_type[:-1]} to update."
                    else:
                        result = await update_person_slack_ids(
                            person_type=person_type,
                            person_name=person_name,
                            slack_user_id=slack_user_id,
                            slack_channel_id=slack_channel_id
                        )
                        if result["success"]:
                            answer = f"✅ {result['message']}\n\n"
                            
                            # Show what was updated
                            if slack_user_id:
                                answer += f"• Slack User ID: `{slack_user_id}`\n"
                            if slack_channel_id:
                                answer += f"• Slack Channel ID: `{slack_channel_id}`\n"
                            
                            # Provide next steps
                            if person_type == "reviewer":
                                answer += "\nThe reviewer can now receive video approval notifications."
                            else:
                                answer += f"\n{person_name} is now integrated with Slack and can receive notifications."
                        else:
                            answer = f"❌ Failed to update Slack IDs: {result['error']}"
            
            elif func_name == "delete_task":
                # Check permissions for deleting tasks
                has_permission, error_msg = simple_check_permission(user_id, "delete_task")
                if not has_permission:
                    answer = error_msg
                else:
                    task_number = args.get("task_number")
                    
                    # Get the task data first to show what we're deleting
                    task_data = await get_task_by_number(task_number)
                    
                    if task_data:
                        # Enter delete confirmation mode
                        pending_deletes[user_id] = {
                            "task_number": task_number,
                            "task_data": task_data
                        }
                        
                        answer = f"⚠️ **Delete Confirmation for Task #{task_number}**\n\n"
                        answer += "**You are about to delete the following task:**\n"
                        answer += f"• Brand: {task_data.get('Brand', 'N/A')}\n"
                        answer += f"• Reference: {task_data.get('Reference Number', 'N/A')}\n"
                        answer += f"• Location: {task_data.get('Location', 'N/A')}\n"
                        answer += f"• Campaign: {task_data.get('Campaign Start Date', 'N/A')} to {task_data.get('Campaign End Date', 'N/A')}\n"
                        answer += f"• Status: {task_data.get('Status', 'N/A')}\n"
                        
                        if str(task_data.get('Status', '')).startswith('Assigned to'):
                            answer += f"• Videographer: {task_data.get('Videographer', 'N/A')}\n"
                            answer += "\n⚠️ **Note:** This task has been assigned and has a Trello card that will be archived.\n"
                        
                        answer += "\n**Are you sure you want to delete this task?**\n"
                        answer += "• Say **'yes'** or **'confirm'** to delete permanently\n"
                        answer += "• Say **'no'** or **'cancel'** to keep the task"
                    else:
                        answer = f"I couldn't find Task #{task_number}. Please check the task number and try again."
            
            else:
                answer = "I encountered an unknown command. Please try again or let me know what you'd like to do."
        else:
            # Regular text response
            text = msg.content[-1].text
            answer = text

        # Store conversation history
        append_to_history(user_id, "user", user_input)
        append_to_history(user_id, "assistant", answer)

        logger.debug(f"\n💬 Assistant Final Reply:\n{answer}\n")

        # Check if this was from a slash command and we have a response URL
        if user_id in slash_command_responses:
            response_url = slash_command_responses[user_id]
            del slash_command_responses[user_id]  # Clean up after use
            
            # Send response using the slash command response URL
            response_data = {
                "response_type": "in_channel",  # Make it visible to everyone
                "text": markdown_to_slack(answer)
            }
            
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    response_url,
                    json=response_data
                )
                response.raise_for_status()
                logger.info(f"✅ Sent slash command response to {response_url}")
            except Exception as e:
                logger.error(f"❌ Failed to send slash command response: {e}")
                # Fall back to regular message
                await slack_client.chat_postMessage(channel=channel, text=markdown_to_slack(answer))
        else:
            # Send regular response to Slack
            try:
                await slack_client.chat_postMessage(channel=channel, text=markdown_to_slack(answer))
            except Exception as e:
                if "channel_not_found" in str(e):
                    # Bot not in channel, send DM instead
                    try:
                        await slack_client.chat_postMessage(channel=user_id, text=markdown_to_slack(
                            f"📨 *Response sent via DM (I'm not in that channel)*\n\n{answer}\n\n"
                            "_To use me in channels, please invite me first._"
                        ))
                    except:
                        logger.error(f"Failed to send DM to user {user_id}")
                else:
                    raise

    except Exception as e:
        error_msg = f"I'm sorry, but I encountered an error processing your request: {str(e)}"
        logger.error(f"Error in main_llm_loop: {e}")
        
        # Try to send error message to channel, fall back to DM if bot not in channel
        try:
            await slack_client.chat_postMessage(channel=channel, text=markdown_to_slack(error_msg))
        except Exception as channel_error:
            if "channel_not_found" in str(channel_error):
                # Send DM to user instead
                try:
                    await slack_client.chat_postMessage(channel=user_id, text=markdown_to_slack(
                        f"I apologize, but I couldn't respond in that channel because I'm not a member. {error_msg}\n\n"
                        "Please use this command in a channel where I'm invited, or invite me to that channel."
                    ))
                except:
                    logger.error(f"Failed to send error message to user {user_id}")


