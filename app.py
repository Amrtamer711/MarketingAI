import asyncio
import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
import requests
import uvicorn
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

from clients import slack_client, signature_verifier, api, logger
from config import UAE_TZ, SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET, OPENAI_API_KEY
from excel_utils import save_to_excel, read_excel_async, initialize_excel
from history import pending_confirmations
from llm_utils import main_llm_loop
from utils import EmailParseRequest, RequestFilter

# Distributed idempotency (Redis) + local fallback
try:
    import redis  # type: ignore
except Exception:
    redis = None

REDIS_URL = os.getenv("REDIS_URL")
_redis = None
if redis and REDIS_URL:
    try:
        _redis = redis.Redis.from_url(REDIS_URL)
    except Exception:
        _redis = None

# Local fallback cache
_PROCESSED_EVENT_KEYS = {}
_DEDUP_TTL_SECONDS = 600

# Burst counter (debug visibility only)
_recent_event_times = defaultdict(deque)
_BURST_WINDOW_SECONDS = 1.5


def should_process_event(key: str, ttl: int = _DEDUP_TTL_SECONDS) -> bool:
    """Return True only the first time a key is seen within ttl across all instances (if Redis configured)."""
    if not key:
        return True
    # Prefer Redis for cross-instance safety
    if _redis is not None:
        try:
            return bool(_redis.set(name=f"slack:evt:{key}", value=1, nx=True, ex=ttl))
        except Exception:
            pass  # fall back to local
    # Local fallback per-process
    now = time.time()
    # purge old
    expired = [k for k, ts in list(_PROCESSED_EVENT_KEYS.items()) if now - ts > ttl]
    for k in expired:
        _PROCESSED_EVENT_KEYS.pop(k, None)
    if key in _PROCESSED_EVENT_KEYS:
        return False
    _PROCESSED_EVENT_KEYS[key] = now
    return True

# Track recent events per channel for quick burst counting
_recent_event_times = defaultdict(deque)
_BURST_WINDOW_SECONDS = 1.5

# Global variable for bot user ID
BOT_USER_ID = None

# ========== LIFESPAN HANDLER ==========

@asynccontextmanager
async def lifespan(app):
    """Initialize resources on startup and clean up on shutdown"""
    global BOT_USER_ID
    
    # Initialize Excel if needed
    await initialize_excel()
    
    if SLACK_BOT_TOKEN:
        try:
            response = await slack_client.auth_test()
            BOT_USER_ID = response["user_id"]
            logger.info(f"✅ Bot User ID initialized on startup: {BOT_USER_ID}")
        except Exception as e:
            logger.error(f"❌ Failed to get bot user ID on startup: {e}")
            logger.error("Bot will not respond to mentions!")
    # Startup completed
    yield
    # Shutdown cleanup - nothing needed anymore

# Register lifespan on existing FastAPI app instance
api.router.lifespan_context = lifespan

# ========== SLACK EVENT HANDLERS ==========
@api.post("/slack/events")
async def slack_events(request: Request):
    """Handle Slack events via HTTP"""
    # Verify the request signature
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")
    
    if not signature_verifier.is_valid(body.decode(), timestamp, signature):
        raise HTTPException(status_code=403, detail="Invalid request signature")
    
    data = await request.json()
    
    # Ignore Slack retries to prevent duplicate processing
    if request.headers.get("X-Slack-Retry-Num"):
        return JSONResponse({"status": "retry_ignored"})
    
    # Handle URL verification challenge
    if data.get("type") == "url_verification":
        return JSONResponse({"challenge": data.get("challenge")})
    
    # Handle events
    if data.get("type") == "event_callback":
        event = data.get("event", {})
        
        # Handle message events - but not app_mention to avoid duplicates
        # app_mention events also trigger message events, so we only need to handle message
        # Also skip button-triggered messages and other subtypes we don't want to process
        subtype = event.get("subtype", "")
        if event.get("type") == "message" and not event.get("bot_id") and subtype not in ["message_changed", "message_deleted", "bot_message"]:
            user_id = event.get("user")
            channel = event.get("channel")
            text = event.get("text", "")
            files = event.get("files", [])
            event_ts = event.get("event_ts", "")
            client_msg_id = event.get("client_msg_id", "")
            
            # Check channel type
            channel_type = await get_channel_type(channel)

            # Burst counter: how many messages in short window for this channel
            now = time.time()
            dq = _recent_event_times[channel]
            dq.append(now)
            # drop old
            while dq and now - dq[0] > _BURST_WINDOW_SECONDS:
                dq.popleft()
            logger.info(f"Burst counter (last {_BURST_WINDOW_SECONDS}s) for channel {channel}: {len(dq)} events")
            
            # Log for debugging
            logger.debug(f"Message received - Channel: {channel}, Type: {channel_type}, User: {user_id}")
            logger.debug(f"Bot User ID: {BOT_USER_ID}, Text: {text[:100]}...")
            logger.debug(f"Event TS: {event_ts}, Client Msg ID: {client_msg_id}, Has files: {len(files) > 0}")
            logger.debug(f"Subtype: {subtype}, Full event: {event}")
            
            # In group channels, only respond if mentioned
            if channel_type in ["channel", "group", "mpim"]:
                is_mentioned = await is_bot_mentioned(text)
                logger.debug(f"Bot mentioned: {is_mentioned}")
                
                if not is_mentioned:
                    # Not mentioned in a group, ignore message
                    logger.debug(f"Ignoring message in {channel_type} - bot not mentioned")
                    return JSONResponse({"status": "ok"})
                else:
                    # Remove the mention from the text for cleaner processing
                    if BOT_USER_ID:
                        # Remove both mention formats
                        text = text.replace(f"<@{BOT_USER_ID}>", "").strip()
                        text = text.replace(f"<!@{BOT_USER_ID}>", "").strip()
            
            # Check if there are file attachments
            if files:
                # Check for video files first
                video_files = [f for f in files if f.get("mimetype", "").startswith("video/")]
                
                if video_files or (files and text and any(word in text.lower() for word in ['task', '#'])):
                    # Dedupe uploads across distributed instances
                    team_id = data.get("team_id", "")
                    file_ids = ",".join(sorted([f.get("id", "") for f in files]))
                    dedupe_key = f"{team_id}:{channel}:{event_ts}:{file_ids or 'nofiles'}"
                    if not should_process_event(dedupe_key):
                        logger.info(f"Duplicate upload ignored: {dedupe_key}")
                        return JSONResponse({"status": "duplicate_ignored"})
                    
                    # Handle video/photo upload with task number
                    try:
                        # Import the new handler
                        from video_upload_system import handle_video_upload_with_parsing
                        
                        # Get the file (video or image)
                        file_to_upload = video_files[0] if video_files else files[0]
                        
                        # Debug logging
                        logger.info(f"Processing video upload - User: {user_id}, Channel: {channel}, Text: '{text}', File: {file_to_upload.get('name', 'unknown')}")
                        
                        # Process the upload (will parse task number from message)
                        await handle_video_upload_with_parsing(channel, user_id, file_to_upload, text)
                        
                    except Exception as e:
                        logger.error(f"Error handling video upload: {e}")
                        await slack_client.chat_postMessage(
                            channel=channel,
                            text="❌ Error processing video upload. Please try again."
                        )
                    
                    return JSONResponse({"status": "ok"})
                
                # Process image attachments
                for file in files:
                    if file.get("mimetype", "").startswith("image/"):
                        # Process image through main LLM loop
                        # Add image context to the message
                        image_message = text if text else "I've uploaded an image with a design request. Please help me extract the details."
                        # Pass the file to main LLM loop
                        asyncio.create_task(main_llm_loop(channel, user_id, image_message, [file]))
                        # Return immediately to avoid Slack timeout
                        return JSONResponse({"status": "ok"})
            
            # Process regular text message
            # Skip if this looks like it might be from a button interaction
            # (empty text and no files typically means button click)
            if text.strip() or files:
                asyncio.create_task(main_llm_loop(channel, user_id, text))
            else:
                logger.debug(f"Skipping empty message with no files - likely button interaction")
            
            # Return immediately to avoid Slack timeout
            return JSONResponse({"status": "ok"})
    
    return JSONResponse({"status": "ok"})

@api.post("/slack/slash-commands")
async def slack_slash_commands(request: Request):
    """Handle Slack slash commands"""
    # Verify the request signature
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")
    
    if not signature_verifier.is_valid(body.decode(), timestamp, signature):
        raise HTTPException(status_code=403, detail="Invalid request signature")
    
    # Parse form data
    form_data = await request.form()
    command = form_data.get("command")
    user_id = form_data.get("user_id")
    channel_id = form_data.get("channel_id")
    
    if command == "/log_campaign" or command == "/design_request":
        # IMPORTANT: Slack slash commands do NOT support file attachments
        # This command only shows documentation on how to log design requests
        help_text = ("📋 *How to Log a Design Request:*\n\n"
                    "*Simply send me a message with one of these formats:*\n\n"
                    "**Option 1 - Manual Entry:**\n"
                    "Type: `I need to log a design request for Brand: Nike, Date: 2024-01-15, Reference: NK-001, Location: Dubai`\n\n"
                    "**Option 2 - Email Content:**\n"
                    "Copy and paste your email content directly to me\n\n"
                    "**Option 3 - Image Upload:**\n"
                    "Upload an image (screenshot, document) directly to me and I'll extract the details\n\n"
                    "**Option 4 - Guided Process:**\n"
                    "Just say: \"I need to log a design request\" and I'll guide you through it\n\n"
                    "⚠️ *Note: Do NOT use slash commands for data input - just send me a regular message!*")
        
        return JSONResponse({
            "response_type": "ephemeral",
            "text": help_text
        })
    
    elif command == "/recent_requests":
        # This command only shows documentation
        help_text = ("📋 *How to View Recent Requests:*\n\n"
                    "Simply send me a message saying:\n"
                    "• \"Show me recent requests\"\n"
                    "• \"What are the recent design requests?\"\n"
                    "• \"List recent tasks\"\n\n"
                    "I'll show you the most recent design requests with their details.\n\n"
                    "⚠️ *Note: Slash commands are for help only - send me a regular message to see the requests!*")
        
        return JSONResponse({
            "response_type": "ephemeral",
            "text": help_text
        })
    
    elif command == "/help_design":
        help_message = ("🎨 *Design Request Bot Help*\n\n"
                       "*Available Documentation Commands:*\n"
                       "• `/help_design` - Show this help menu\n"
                       "• `/log_campaign` - Learn how to log design requests\n"
                       "• `/recent_requests` - Learn how to view recent requests\n"
                       "• `/my_ids` - Show your Slack user and channel IDs\n\n"
                       "*How to Use This Bot:*\n"
                       "⚠️ **Important**: Slash commands are for documentation only!\n"
                       "To actually perform actions, send me regular messages:\n\n"
                       "**To Log a Request:**\n"
                       "• Manual: \"Log request for Brand: Nike, Date: 2024-01-15, Reference: NK-001\"\n"
                       "• Email: Just paste the email content\n"
                       "• Image: Upload a screenshot or document\n\n"
                       "**To View Requests:**\n"
                       "• \"Show me recent requests\"\n"
                       "• \"List all tasks\"\n\n"
                       "**To Edit Tasks:**\n"
                       "• \"Edit task 123\"\n"
                       "• \"Update task 45\"\n\n"
                       "*Remember: Just talk to me naturally - no slash commands needed for actions!*")
        
        return JSONResponse({
            "response_type": "ephemeral",
            "text": help_message
        })
    
    elif command == "/my_ids":
        # Get user info
        try:
            user_info = await slack_client.users_info(user=user_id)
            user_name = user_info["user"]["profile"].get("real_name", "Unknown")
            
            # Get channel info
            channel_type = "Unknown"
            channel_name = "Unknown"
            try:
                channel_info = await slack_client.conversations_info(channel=channel_id)
                if channel_info["ok"]:
                    chan = channel_info["channel"]
                    channel_name = chan.get("name", "Direct Message")
                    if chan.get("is_channel"):
                        channel_type = "Public Channel"
                    elif chan.get("is_group"):
                        channel_type = "Private Channel"
                    elif chan.get("is_im"):
                        channel_type = "Direct Message"
                    elif chan.get("is_mpim"):
                        channel_type = "Group DM"
            except:
                pass
            
            # Get email from user profile
            user_email = user_info["user"]["profile"].get("email", "Not available")
            
            id_message = (f"🆔 *Your Slack Information*\n\n"
                         f"*User Details:*\n"
                         f"• Name: {user_name}\n"
                         f"• Email: {user_email}\n"
                         f"• User ID: `{user_id}`\n\n"
                         f"*Channel Information:*\n"
                         f"• Channel: {channel_name}\n"
                         f"• Type: {channel_type}\n"
                         f"• Channel ID: `{channel_id}`\n\n"
                         f"📋 *Copyable Format for Admin:*\n"
                         f"```\n"
                         f"Name: {user_name}\n"
                         f"Email: {user_email}\n"
                         f"Slack User ID: {user_id}\n"
                         f"Slack Channel ID: {channel_id}\n"
                         f"```\n\n"
                         f"💡 *Next Steps:*\n"
                         f"1. Copy the above information\n"
                         f"2. Send it to your admin (Head of Department or Reviewer)\n"
                         f"3. They will add you to the system with these IDs")
            
            return JSONResponse({
                "response_type": "ephemeral",
                "text": id_message
            })
            
        except Exception as e:
            logger.error(f"Error getting IDs: {e}")
            return JSONResponse({
                "response_type": "ephemeral",
                "text": f"❌ Error getting your IDs: {str(e)}"
            })
    
    elif command == "/upload_video":
        # This command only shows documentation
        help_text = ("📹 *How to Upload Videos:*\n\n"
                    "**For Videographers Only:**\n"
                    "1. Simply upload your video file directly to me\n"
                    "2. Include the task number in your message (e.g., \"Task 123\" or just \"123\")\n"
                    "3. I'll ask you to choose between:\n"
                    "   • **Raw** - For videos meeting the deadline\n"
                    "   • **Pending** - For videos ready for review\n\n"
                    "**Example:**\n"
                    "Upload your video with a message like: \"Here's the video for task 123\"\n\n"
                    "⚠️ *Note: Only registered videographers can upload videos.*\n"
                    "⚠️ *Slash commands cannot accept file uploads - just send me the video directly!*")
        
        return JSONResponse({
            "response_type": "ephemeral",
            "text": help_text
        })
    
    return JSONResponse({"response_type": "ephemeral", "text": "Unknown command. Use `/help_design` for available commands."})

# ========== SLACK INTERACTIVE COMPONENTS ==========
# Button click tracking to prevent spam
_button_clicks = defaultdict(lambda: defaultdict(float))  # user_id -> action_id -> timestamp
DEBOUNCE_WINDOW_SECONDS = 3  # 3 second debounce window

def is_button_spam(user_id: str, action_id: str, value: str = None) -> bool:
    """Check if a button click is spam based on debouncing window"""
    current_time = time.time()
    # Create a unique key including the value for actions with specific values
    action_key = f"{action_id}:{value}" if value else action_id
    last_click = _button_clicks[user_id][action_key]
    
    if current_time - last_click < DEBOUNCE_WINDOW_SECONDS:
        return True
    
    _button_clicks[user_id][action_key] = current_time
    return False

@api.post("/slack/interactive")
async def slack_interactive(request: Request):
    """Handle Slack interactive components (buttons, select menus, etc.)"""
    # Verify the request signature
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")
    
    if not signature_verifier.is_valid(body.decode(), timestamp, signature):
        raise HTTPException(status_code=403, detail="Invalid request signature")
    
    # Parse the payload
    form_data = await request.form()
    payload = json.loads(form_data.get("payload", "{}"))
    
    # Handle different interaction types
    interaction_type = payload.get("type")
    
    if interaction_type == "block_actions":
        # Handle button clicks
        user_id = payload["user"]["id"]
        actions = payload.get("actions", [])
        response_url = payload.get("response_url")
        channel = payload["channel"]["id"]
        
        for action in actions:
            action_id = action.get("action_id")
            action_value = action.get("value", "")
            
            # Check for spam clicks
            if is_button_spam(user_id, action_id, action_value):
                logger.info(f"Spam click detected from {user_id} on {action_id}")
                # Send ephemeral message about the debounce
                if response_url:
                    requests.post(response_url, json={
                        "replace_original": False,
                        "response_type": "ephemeral",
                        "text": "⏳ Please wait a moment before clicking again..."
                    })
                return JSONResponse({"ok": True})
            
            if action_id in ["approve_video", "reject_video"]:
                # Import and use the video upload system
                from video_upload_system import handle_approval_action
                
                # Process the approval/rejection
                asyncio.create_task(
                    handle_approval_action(action, user_id, response_url)
                )
                
                # Send immediate response
                return JSONResponse({"text": "Processing your action..."})
            
            elif action_id in ["select_raw_folder", "select_pending_folder"]:
                # Handle video upload folder selection
                try:
                    value_data = json.loads(action["value"])
                    folder = value_data["folder"]
                    file_id = value_data["file_id"]
                    file_name = value_data["file_name"]
                    action_type = value_data.get("action", "upload_video")
                    
                    # Check if this is a task-based upload
                    if action_type == "upload_video_by_task":
                        task_number = value_data.get("task_number")
                        
                        # Import video upload system
                        from video_upload_system import handle_video_upload_by_task_number
                        
                        # Get file info
                        file_info = {
                            "id": file_id,
                            "name": file_name
                        }
                        
                        # Process the upload with task number
                        asyncio.create_task(
                            handle_video_upload_by_task_number(channel, user_id, file_info, task_number, folder)
                        )
                        
                        # Update the message
                        requests.post(response_url, json={
                            "replace_original": True,
                            "text": f"🎥 Processing upload for Task #{task_number} to {folder} folder..."
                        })
                    else:
                        # Legacy upload without task number
                        from video_upload_system import handle_video_upload
                        
                        # Get file info
                        file_info = {
                            "id": file_id,
                            "name": file_name
                        }
                        
                        # Process the upload
                        asyncio.create_task(
                            handle_video_upload(channel, user_id, file_info, folder)
                        )
                        
                        # Update the message
                        requests.post(response_url, json={
                            "replace_original": True,
                            "text": f"🎥 Processing upload of `{file_name}` to {folder} folder..."
                        })
                    
                except Exception as e:
                    logger.error(f"Error processing video folder selection: {e}")
                    requests.post(response_url, json={
                        "replace_original": True,
                        "text": "❌ Error processing video upload. Please try again."
                    })
                
                return JSONResponse({"ok": True})
            
            # Handle video approval workflow actions
            elif action_id in ["approve_video_reviewer", "reject_video_reviewer", "approve_video_hos", "reject_video_hos"]:
                workflow_id = action.get("value")
                
                from video_upload_system import (
                    handle_reviewer_approval, handle_reviewer_rejection,
                    handle_hos_approval, handle_hos_rejection
                )
                
                if action_id == "approve_video_reviewer":
                    # Send immediate "Please wait" response
                    requests.post(response_url, json={
                        "replace_original": True,
                        "text": "⏳ Please wait... Processing approval..."
                    })
                    asyncio.create_task(handle_reviewer_approval(workflow_id, user_id, response_url))
                elif action_id == "reject_video_reviewer":
                    # Open modal for rejection comments
                    from video_upload_system import approval_workflows
                    workflow = approval_workflows.get(workflow_id, {})
                    task_number = workflow.get('task_number', 'Unknown')
                    
                    await slack_client.views_open(
                        trigger_id=payload["trigger_id"],
                        view={
                            "type": "modal",
                            "callback_id": f"reject_video_modal_{workflow_id}",
                            "title": {"type": "plain_text", "text": "Reject Video"},
                            "blocks": [
                                {
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": f"Please provide a reason for rejecting the video for Task #{task_number}"
                                    }
                                },
                                {
                                    "type": "input",
                                    "block_id": "rejection_reason",
                                    "label": {"type": "plain_text", "text": "Rejection Reason"},
                                    "element": {
                                        "type": "plain_text_input",
                                        "action_id": "reason_input",
                                        "multiline": True,
                                        "placeholder": {"type": "plain_text", "text": "Enter the reason for rejection..."}
                                    }
                                }
                            ],
                            "submit": {"type": "plain_text", "text": "Submit"},
                            "close": {"type": "plain_text", "text": "Cancel"},
                            "private_metadata": json.dumps({
                                "workflow_id": workflow_id,
                                "response_url": response_url,
                                "stage": "reviewer"
                            })
                        }
                    )
                elif action_id == "approve_video_hos":
                    # Send immediate "Please wait" response
                    requests.post(response_url, json={
                        "replace_original": True,
                        "text": "⏳ Please wait... Processing final approval..."
                    })
                    asyncio.create_task(handle_hos_approval(workflow_id, user_id, response_url))
                elif action_id == "reject_video_hos":
                    # Open modal for rejection comments
                    from video_upload_system import approval_workflows
                    workflow = approval_workflows.get(workflow_id, {})
                    task_number = workflow.get('task_number', 'Unknown')
                    
                    await slack_client.views_open(
                        trigger_id=payload["trigger_id"],
                        view={
                            "type": "modal",
                            "callback_id": f"reject_video_modal_{workflow_id}",
                            "title": {"type": "plain_text", "text": "Reject Video"},
                            "blocks": [
                                {
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": f"Please provide a reason for rejecting the video for Task #{task_number}"
                                    }
                                },
                                {
                                    "type": "input",
                                    "block_id": "rejection_reason",
                                    "label": {"type": "plain_text", "text": "Rejection Reason"},
                                    "element": {
                                        "type": "plain_text_input",
                                        "action_id": "reason_input",
                                        "multiline": True,
                                        "placeholder": {"type": "plain_text", "text": "Enter the reason for rejection..."}
                                    }
                                }
                            ],
                            "submit": {"type": "plain_text", "text": "Submit"},
                            "close": {"type": "plain_text", "text": "Cancel"},
                            "private_metadata": json.dumps({
                                "workflow_id": workflow_id,
                                "response_url": response_url,
                                "stage": "hos"
                            })
                        }
                    )
                
                return JSONResponse({"text": "Processing..."})
    
    elif interaction_type == "view_submission":
        # Handle modal submissions
        callback_id = payload["view"]["callback_id"]
        user_id = payload["user"]["id"]
        
        # Check for spam modal submissions
        if is_button_spam(user_id, "modal_submission", callback_id):
            logger.info(f"Spam modal submission detected from {user_id} on {callback_id}")
            return JSONResponse({
                "response_action": "errors",
                "errors": {
                    "rejection_reason": "Please wait a moment before submitting again..."
                }
            })
        
        if callback_id.startswith("reject_video_modal_"):
            # Handle rejection modal submission
            metadata = json.loads(payload["view"]["private_metadata"])
            workflow_id = metadata["workflow_id"]
            response_url = metadata["response_url"]
            stage = metadata["stage"]
            
            # Get rejection reason from modal
            rejection_reason = payload["view"]["state"]["values"]["rejection_reason"]["reason_input"]["value"]
            
            from video_upload_system import handle_reviewer_rejection, handle_hos_rejection
            
            # Send immediate "Please wait" response
            requests.post(response_url, json={
                "replace_original": True,
                "text": "⏳ Please wait... Processing rejection..."
            })
            
            # Process rejection based on stage
            if stage == "reviewer":
                asyncio.create_task(handle_reviewer_rejection(workflow_id, user_id, response_url, rejection_reason))
            elif stage == "hos":
                asyncio.create_task(handle_hos_rejection(workflow_id, user_id, response_url, rejection_reason))
            
            return JSONResponse({"response_action": "clear"})
    
    return JSONResponse({"ok": True})

# ========== FASTAPI ENDPOINTS ==========
@api.post("/api/parse_email")
async def api_parse_email(request: EmailParseRequest):
    """API endpoint to parse email content using AI"""
    try:
        # Use main LLM loop to parse email
        # Create a temporary user ID for API requests
        temp_user_id = f"api_{datetime.now(UAE_TZ).timestamp()}"
        
        # Process through main LLM loop
        await main_llm_loop(
            channel="api",
            user_id=temp_user_id,
            user_input=f"Please log this design request from email: {request.email_text}"
        )
        
        # Check if there's a pending confirmation
        if temp_user_id in pending_confirmations:
            parsed_data = pending_confirmations[temp_user_id]
            del pending_confirmations[temp_user_id]
            
            # Optionally save to Excel if requested
            if request.save_to_excel:
                parsed_data["submitted_by"] = request.submitted_by
                result = await save_to_excel(parsed_data)
                if not result["success"]:
                    raise HTTPException(status_code=500, detail="Failed to save to Excel")
                parsed_data["task_number"] = result["task_number"]
            
            return JSONResponse({
                "success": True,
                "parsed_data": parsed_data
            })
        else:
            return JSONResponse({
                "success": False,
                "message": "Could not parse email. Required fields may be missing."
            })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/api/get_requests")
async def api_get_requests():
    """API endpoint to retrieve all requests"""
    try:
        df = await read_excel_async()
        # Convert to JSON, handling any datetime objects
        requests = df.to_dict(orient="records")
        return JSONResponse({
            "success": True,
            "requests": requests,
            "count": len(requests)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.post("/api/export_requests")
async def api_export_requests(filters: RequestFilter):
    """API endpoint to export requests with filters"""
    try:
        df = await read_excel_async()
        
        # Apply filters asynchronously
        if filters.start_date:
            df = df[df["Campaign Start Date"] >= filters.start_date]
        if filters.end_date:
            df = df[df["Campaign End Date"] <= filters.end_date]
        if filters.brand:
            df = df[df["Brand"].str.contains(filters.brand, case=False, na=False)]
        
        # Convert to JSON
        filtered_requests = df.to_dict(orient="records")
        
        return JSONResponse({
            "success": True,
            "requests": filtered_requests,
            "count": len(filtered_requests),
            "filters_applied": {
                "start_date": filters.start_date,
                "end_date": filters.end_date,
                "brand": filters.brand
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now(UAE_TZ).isoformat()}

@api.get("/dashboard")
async def dashboard():
    html = """
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
      <title>Video Dashboard</title>
      <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
      <script src=\"https://cdn.tailwindcss.com\"></script>
      <style>
        body { background:#000; color:#fff; }
        .card { background:#111827; box-shadow: 0 1px 2px rgba(0,0,0,0.6); border-radius: 0.5rem; padding: 1rem; }
        .chip { display:inline-block; padding:0.25rem 0.5rem; border-radius: 0.25rem; font-size:0.75rem; font-weight:500; background:#1f2937; color:#e5e7eb; margin-right:0.5rem; margin-bottom:0.5rem; }
        .btn { padding:0.25rem 0.75rem; border-radius: 0.375rem; background:#2563eb; color:white; font-size:0.875rem; }
        .btn:hover { background:#1d4ed8; }
        .btn-toggle { padding:0.25rem 0.75rem; border-radius: 0.375rem; border:1px solid #6b7280; color:#e5e7eb; font-size:0.875rem; }
        .btn-active { background:#e5e7eb; color:#111827; border-color:#e5e7eb; }
        .btn-outline { padding:0.5rem 1rem; border-radius:0.5rem; border:1px solid #e5e7eb; color:#e5e7eb; font-size:1rem; }
        .btn-outline:hover { background:#374151; }
        details > summary { cursor: pointer; }
        .no-scrollbar::-webkit-scrollbar { display: none; }
        .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
        .summary-table { width: 100%; table-layout: fixed; font-size: 1.05rem; }
        .summary-table th, .summary-table td { white-space: normal; word-break: break-word; padding: 12px 16px; }
      </style>
    </head>
    <body class=\"bg-black text-white\">
      <div class=\"max-w-screen-2xl mx-auto p-6 space-y-6\">
        <div class=\"flex items-center justify-between\">
          <h1 class=\"text-2xl font-semibold\">HOD Video Dashboard</h1>
          <div class=\"flex items-center gap-2\">
            <div class=\"space-x-2\">
              <button id=\"mMonth\" class=\"btn-toggle\" onclick=\"setMode('month')\">Month</button>
              <button id=\"mYear\" class=\"btn-toggle\" onclick=\"setMode('year')\">Year</button>
            </div>
            <button id=\"calendarBtn\" class=\"btn flex items-center gap-2\" onclick=\"openCalendar()\">
              <svg class=\"w-5 h-5\" fill=\"none\" stroke=\"currentColor\" viewBox=\"0 0 24 24\">
                <path stroke-linecap=\"round\" stroke-linejoin=\"round\" stroke-width=\"2\" d=\"M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z\"></path>
              </svg>
              <span id=\"selectedPeriod\">August 2025</span>
            </button>
            <input id=\"pMonth\" type=\"month\" class=\"border rounded p-2 bg-black text-white border-gray-600 hidden\" />
            <input id=\"pYear\" type=\"number\" min=\"2000\" max=\"2100\" class=\"border rounded p-2 bg-black text-white border-gray-600 hidden w-24\" />
            <button class=\"btn\" onclick=\"loadData()\">Apply</button>
          </div>
        </div>

        <div class=\"grid grid-cols-1 md:grid-cols-3 gap-6\">
          <div class=\"card col-span-1\">
            <h2 class=\"font-medium mb-2\">Completed vs Not Completed</h2>
            <canvas id=\"assignPie\" height=\"200\"></canvas>
            <div id=\"assignLegend\" class=\"text-sm text-gray-300 mt-3\"></div>
          </div>
          <div class=\"card col-span-2\">
            <h2 class=\"font-medium mb-2\">Summary</h2>
            <div id=\"summary\" class=\"text-sm text-gray-200\"></div>
          </div>
        </div>

        <div class=\"card\">
          <h2 class=\"font-medium mb-2\">Reviewer Summary</h2>
          <div id=\"reviewerBlock\"></div>
        </div>

        <div class=\"card\">
          <h2 class=\"font-medium mb-4\">Per-Videographer Analysis</h2>
          <div id=\"videographers\" class=\"space-y-6\"></div>
        </div>
      </div>

      <script>
        let mode = 'month';
        
        function setMode(m) {
          mode = m;
          document.querySelectorAll('.btn-toggle').forEach(b => b.classList.remove('btn-active'));
          document.getElementById(m === 'year' ? 'mYear' : 'mMonth').classList.add('btn-active');
          updateSelectedPeriodDisplay();
          loadData();
        }
        
        function currentPeriod() {
          if (mode === 'year') return document.getElementById('pYear').value;
          return document.getElementById('pMonth').value;
        }
        
        function openCalendar() {
          const input = document.getElementById(mode === 'year' ? 'pYear' : 'pMonth');
          if (mode === 'year') {
            // For year, show a simple prompt since number inputs don't have a picker
            const year = prompt('Enter year (2000-2100):', input.value);
            if (year && year >= 2000 && year <= 2100) {
              input.value = year;
              updateSelectedPeriodDisplay();
              loadData();
            }
          } else {
            // For month, trigger the native date picker
            try {
              input.showPicker();
            } catch(e) {
              input.click();
            }
          }
        }
        
        function updateSelectedPeriodDisplay() {
          const period = currentPeriod();
          const display = document.getElementById('selectedPeriod');
          
          if (mode === 'year') {
            display.textContent = period || 'Select Year';
          } else {
            if (period) {
              const [year, month] = period.split('-');
              const date = new Date(year, month - 1);
              display.textContent = date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
            } else {
              display.textContent = 'Select Month';
            }
          }
        }

        async function loadData() {
          try {
            const period = currentPeriod();
            const res = await fetch(`/api/dashboard?mode=${mode}&period=${encodeURIComponent(period)}`);
            if (!res.ok) {
              throw new Error(`HTTP error! status: ${res.status}`);
            }
            const data = await res.json();
            setSummaryVg(data.summary_videographers || {});
            renderPie(data.pie || { completed: 0, not_completed: 0 });
            renderSummary(data.summary || {});
            renderReviewer(data.reviewer || {});
            renderVideographers(data.videographers || []);
          } catch (error) {
            console.error('Error loading dashboard data:', error);
            alert('Error loading dashboard data. Please check the console.');
          }
        }

        let pieChart;
        function renderPie(pie) {
          const ctx = document.getElementById('assignPie');
          if (pieChart) pieChart.destroy();
          const completed = pie.completed || 0;
          const notCompleted = pie.not_completed || 0;
          pieChart = new Chart(ctx, {
            type: 'pie',
            data: {
              labels: ['Completed', 'Not Completed'],
              datasets: [{
                data: [completed, notCompleted],
                backgroundColor: ['#16a34a', '#6b7280']
              }]
            },
            options: { plugins: { legend: { position: 'bottom', labels: { color: '#fff' } } } }
          });
          document.getElementById('assignLegend').innerText = `Completed: ${completed} | Not Completed: ${notCompleted}`;
        }

        function renderSummary(summary) {
          const el = document.getElementById('summary');
          const periodLabel = mode === 'day' ? 'Today' : mode === 'year' ? 'This Year' : 'This Month';
          el.innerHTML = `
            <div class=\"grid grid-cols-3 md:grid-cols-5 gap-4\">
              <div class=\"text-center p-3 border border-gray-700 rounded bg-black/30\">
                <div class=\"text-gray-400 text-sm\">Total Videos ${periodLabel}</div>
                <div class=\"text-2xl font-bold mt-1\">${summary.total||0}</div>
              </div>
              <div class=\"text-center p-3 border border-gray-700 rounded bg-black/30\">
                <div class=\"text-gray-400 text-sm\">Number of Uploads</div>
                <div class=\"text-2xl font-bold mt-1\">${summary.uploads||0}</div>
              </div>
              <div class=\"text-center p-3 border border-gray-700 rounded bg-black/30\">
                <div class=\"text-gray-400 text-sm\">Pending Videos</div>
                <div class=\"text-2xl font-bold mt-1\">${summary.pending||0}</div>
              </div>
              <div class=\"text-center p-3 border border-gray-700 rounded bg-black/30\">
                <div class=\"text-gray-400 text-sm\">Rejected Videos</div>
                <div class=\"text-2xl font-bold mt-1\">${summary.rejected||0}</div>
              </div>
              <div class=\"text-center p-3 border border-gray-700 rounded bg-black/30\">
                <div class=\"text-gray-400 text-sm\">Submitted to Sales</div>
                <div class=\"text-2xl font-bold mt-1\">${summary.submitted_to_sales||0}</div>
              </div>
              <div class=\"text-center p-3 border border-gray-700 rounded bg-black/30\">
                <div class=\"text-gray-400 text-sm\">Returned Videos</div>
                <div class=\"text-2xl font-bold mt-1\">${summary.returned||0}</div>
              </div>
              <div class=\"text-center p-3 border border-gray-700 rounded bg-black/30\">
                <div class=\"text-gray-400 text-sm\">Accepted Videos</div>
                <div class=\"text-2xl font-bold mt-1\">${summary.accepted_videos||0}</div>
              </div>
              <div class=\"text-center p-3 border border-gray-700 rounded bg-success/20\">
                <div class=\"text-gray-400 text-sm\">Accepted %</div>
                <div class=\"text-2xl font-bold mt-1 text-green-400\">${summary.accepted_pct||0}%</div>
              </div>
              <div class=\"text-center p-3 border border-gray-700 rounded bg-danger/20\">
                <div class=\"text-gray-400 text-sm\">Rejected %</div>
                <div class=\"text-2xl font-bold mt-1 text-red-400\">${summary.rejected_pct||0}%</div>
              </div>
            </div>`;
        }
        // store per-vg summary for rendering columns
        function setSummaryVg(data){ window._summaryVg = data || {}; }
        
        function renderReviewer(reviewer) {
          const el = document.getElementById('reviewerBlock');
          el.innerHTML = `
            <div class=\"grid grid-cols-1 md:grid-cols-3 gap-4\">
              <div class=\"text-center p-3 border border-gray-700 rounded bg-black/30\">
                <div class=\"text-gray-400 text-sm\">Avg Response Time</div>
                <div class=\"text-2xl font-bold mt-1\">${reviewer.avg_response_display || (reviewer.avg_response_hours ? reviewer.avg_response_hours + ' hrs' : '0 hrs')}</div>
              </div>
              <div class=\"text-center p-3 border border-gray-700 rounded bg-black/30\">
                <div class=\"text-gray-400 text-sm\">Pending Videos</div>
                <div class=\"text-2xl font-bold mt-1\">${reviewer.pending_videos||0}</div>
              </div>
              <div class=\"text-center p-3 border border-gray-700 rounded bg-black/30\">
                <div class=\"text-gray-400 text-sm\">Handled Success Rate</div>
                <div class=\"text-2xl font-bold mt-1\">${reviewer.handled_percent||0}%</div>
              </div>
            </div>`;
        }

        function renderVideographers(vmap) {
          const el = document.getElementById('videographers');
          el.innerHTML = '';
          const vgSummary = window._summaryVg || {};
          
          Object.keys(vmap).sort().forEach(v => {
            const tasks = vmap[v];
            const stats = vgSummary[v] || {};
            const wrap = document.createElement('div');
            wrap.className = 'p-4 rounded border border-gray-700 bg-black/20';
            
            // Header with stats columns
            wrap.innerHTML = `
              <div class=\"mb-4\">
                <div class=\"text-lg font-medium mb-3\">${v}</div>
                <div class=\"grid grid-cols-2 md:grid-cols-7 gap-3\">
                  <div class=\"text-center p-2 border border-gray-700 rounded bg-black/30\">
                    <div class=\"text-gray-400 text-xs\">Total Tasks</div>
                    <div class=\"text-xl font-bold mt-1\">${stats.total||0}</div>
                  </div>
                  <div class=\"text-center p-2 border border-gray-700 rounded bg-black/30\">
                    <div class=\"text-gray-400 text-xs\">Uploads</div>
                    <div class=\"text-xl font-bold mt-1\">${stats.uploads||0}</div>
                  </div>
                  <div class=\"text-center p-2 border border-gray-700 rounded bg-black/30\">
                    <div class=\"text-gray-400 text-xs\">Pending</div>
                    <div class=\"text-xl font-bold mt-1\">${stats.pending||0}</div>
                  </div>
                  <div class=\"text-center p-2 border border-gray-700 rounded bg-black/30\">
                    <div class=\"text-gray-400 text-xs\">Rejected</div>
                    <div class=\"text-xl font-bold mt-1\">${stats.rejected||0}</div>
                  </div>
                  <div class=\"text-center p-2 border border-gray-700 rounded bg-black/30\">
                    <div class=\"text-gray-400 text-xs\">In Sales</div>
                    <div class=\"text-xl font-bold mt-1\">${stats.submitted_to_sales||0}</div>
                  </div>
                  <div class=\"text-center p-2 border border-gray-700 rounded bg-black/30\">
                    <div class=\"text-gray-400 text-xs\">Accepted</div>
                    <div class=\"text-xl font-bold mt-1\">${stats.accepted_videos||0}</div>
                  </div>
                  <div class=\"text-center p-2 border border-gray-700 rounded bg-success/20\">
                    <div class=\"text-gray-400 text-xs\">Success Rate</div>
                    <div class=\"text-xl font-bold mt-1 text-green-400\">${stats.accepted_pct||0}%</div>
                  </div>
                </div>
              </div>`;
            
            const list = document.createElement('div');
            list.className = 'mt-3 space-y-3';
            tasks.forEach(t => {
              const card = document.createElement('div');
              card.className = 'p-3 rounded border border-gray-700 bg-black/30';
              const versions = (t.versions || []).map(ver => `
                <details class=\"mt-2\">
                  <summary class=\"text-sm\">Version ${ver.version}</summary>
                  <div class=\"mt-2 text-sm text-gray-200\">${(ver.lifecycle || []).map(item => {
                    let html = `<div class=\"chip\">${item.stage}: ${item.at}`;
                    if (item.rejection_class) {
                      html += ` - ${item.rejection_class}`;
                      if (item.rejected_by) html += ` by ${item.rejected_by}`;
                      if (item.rejection_comments) html += ` (${item.rejection_comments})`;
                    }
                    html += `</div>`;
                    return html;
                  }).join('') || 'No events'}</div>
                </details>`).join('');
              card.innerHTML = `
                <div class=\"flex flex-nowrap items-center justify-between gap-4\"> 
                  <div>
                    <div class=\"font-medium\">Task #${t.task_number} — ${t.brand}</div>
                    <div class=\"text-sm text-gray-300\">Ref: ${t.reference_number || 'NA'}</div>
                  </div>
                  <div class=\"text-sm text-gray-200 whitespace-nowrap overflow-x-auto no-scrollbar\"> 
                    <span class=\"chip\">Filming Deadline: ${t.filming_date || 'NA'}</span>
                    <span class=\"chip\">Uploaded Last Version: ${t.submitted_at || 'NA'}</span>
                    <span class=\"chip\">Current Version Number: ${t.current_version || 'NA'}</span>
                    <span class=\"chip\">Submitted to Sales at: ${t.submitted_to_sales_at || 'NA'}</span>
                    <span class=\"chip\">Accepted at: ${t.accepted_at || 'NA'}</span>
                  </div>
                </div>
                <div class=\"mt-2\">${versions}</div>
              `;
              list.appendChild(card);
            });
            wrap.appendChild(list);
            el.appendChild(wrap);
          });
        }

        // initial load
        (function(){
          // Set default values
          const d = new Date();
          document.getElementById('pMonth').value = d.toISOString().slice(0,7);
          document.getElementById('pYear').value = d.getFullYear();
          
          // Add event listeners to update display when values change
          document.getElementById('pMonth').addEventListener('change', () => {
            updateSelectedPeriodDisplay();
            loadData();
          });
          document.getElementById('pYear').addEventListener('change', () => {
            updateSelectedPeriodDisplay();
            loadData();
          });
          
          setMode('month');
          loadData();
        })();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

@api.get("/api/dashboard")
async def api_dashboard(mode: str = "month", period: str = ""):
    try:
        # Period parsing helpers
        def in_period(d) -> bool:
            if d is None:
                return False
            if mode == 'year':
                y = int((period or datetime.now(UAE_TZ).strftime('%Y')))
                return d.year == y
            # default month: period expects YYYY-MM
            try:
                p = period or datetime.now(UAE_TZ).strftime('%Y-%m')
                y, m = map(int, p.split('-'))
                return d.year == y and d.month == m
            except Exception:
                return False
        
        df = await read_excel_async()
        # Ensure expected columns exist
        for col in [
            'Task #','Brand','Reference Number','Filming Date','Videographer',
            'Current Version','Version History','Pending Timestamps','Submitted Timestamps','Accepted Timestamps','Status'
        ]:
            if col not in df.columns:
                df[col] = ''
        
        # Normalize Filming Date to date
        def parse_date(val):
            try:
                if pd.isna(val) or val == '':
                    return None
                if isinstance(val, str):
                    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
                        try:
                            return datetime.strptime(val, fmt).date()
                        except Exception:
                            continue
                    return None
                return pd.to_datetime(val).date()
            except Exception:
                return None
        
        # Helper to coerce NaN/None/'' to 'NA'
        import math
        def safe_text(val: object) -> str:
            try:
                if val is None:
                    return 'NA'
                if isinstance(val, float) and math.isnan(val):
                    return 'NA'
                s = str(val)
                if s.strip().lower() == 'nan':
                    return 'NA'
                return s if s.strip() != '' else 'NA'
            except Exception:
                return 'NA'
        
        df['__filming_date'] = df['Filming Date'].apply(parse_date)
        
        logger.info(f"\n=== FILTERING BY FILMING DATE ===")
        logger.info(f"Total rows in Excel: {len(df)}")
        logger.info(f"Period mode: {mode}, Period value: {period}")
        
        # Debug filming dates
        for _, row in df.head(10).iterrows():
            fd = row.get('Filming Date', 'NA')
            parsed = row.get('__filming_date')
            task = row.get('Task #', 'NA')
            if parsed:
                check = in_period(parsed)
                logger.info(f"Task #{task}: Filming Date={fd} → Parsed={parsed} → In Period={check}")
            else:
                logger.info(f"Task #{task}: Filming Date={fd} → Could not parse")
        
        # Filter by filming date - tasks scheduled for this period show all their versions
        scope = df[df['__filming_date'].apply(in_period)].copy()
        logger.info(f"Tasks in scope (filtered by filming date): {len(scope)}")
        
        # Handle empty data
        if len(scope) == 0 or len(df) == 0:
            return {
                "period": period,
                "mode": mode,
                "pie": {
                    "completed": 0,
                    "not_completed": 0
                },
                "summary": {
                    "total": 0,
                    "assigned": 0,
                    "pending": 0,
                    "review": 0,
                    "done": 0,
                    "completion_rate": "0%"
                },
                "reviewer": {},
                "videographers": [],
                "summary_videographers": {}
            }
        
        # Calculate metrics based on version history
        total = int(scope.shape[0])
        assigned = int((scope['Videographer'].fillna('') != '').sum())
        
        logger.info(f"\n=== OVERALL METRICS CALCULATION ===")
        logger.info(f"Total tasks in scope: {total}")
        logger.info(f"Assigned tasks: {assigned}")
        logger.info(f"Current period mode: {mode}")
        logger.info(f"Current period value: {period}")
        
        # Count videos from version history
        uploads_cnt = 0  # Total unique versions uploaded
        rejected_cnt = 0  # Total rejected events
        submitted_cnt = 0  # Total submitted to sales events
        returned_cnt = 0  # Total returned events
        accepted_cnt = 0  # Total accepted events
        
        tasks_with_events_in_period = 0
        
        for row_idx, (_, row) in enumerate(scope.iterrows()):
            try:
                task_num = row.get('Task #', 'NA')
                vh = json.loads(str(row.get('Version History') or '[]'))
                if not vh:
                    continue
                    
                events_in_period = 0
                unique_versions = set()  # Track unique versions for uploads
                logger.info(f"\nTask #{task_num}: Checking {len(vh)} events")
                
                for event in vh:
                    event_at = event.get('at', '')
                    version = event.get('version', '?')
                    folder = event.get('folder', '').lower()
                    
                    # Count ALL events for tasks with filming date in period
                    events_in_period += 1
                    logger.info(f"  ✓ v{version} → {folder} at {event_at}")
                    
                    # Track unique versions that went to pending (uploads)
                    if folder == 'pending':
                        unique_versions.add(version)
                    
                    # Count rejection/return/acceptance events
                    if folder == 'rejected':
                        rejected_cnt += 1
                    elif folder == 'submitted to sales':
                        submitted_cnt += 1
                    elif folder == 'returned':
                        returned_cnt += 1
                    elif folder == 'accepted':
                        accepted_cnt += 1
                    else:
                        if folder not in ['pending']:
                            logger.info(f"    Unknown folder: {folder}")
                
                # Add unique version count to uploads
                uploads_cnt += len(unique_versions)
                logger.info(f"  Task #{task_num} has {len(unique_versions)} unique versions uploaded")
                
                if events_in_period > 0:
                    tasks_with_events_in_period += 1
                    logger.info(f"  Task #{task_num} has {events_in_period} events in period")
                    
            except Exception as e:
                logger.error(f"Error processing task {row_idx}: {e}")
        
        # Get CURRENT status counts from ALL tasks (not just in period)
        df_all = await read_excel_async()
        status_all = df_all['Status'].astype(str).str.strip()
        current_pending = int((status_all == 'Critique').sum())
        current_submitted = int((status_all == 'Submitted to Sales').sum())
        
        logger.info(f"\n=== METRICS SUMMARY ===")
        logger.info(f"Tasks with events in period: {tasks_with_events_in_period}")
        logger.info(f"Total Uploads: {uploads_cnt}")
        logger.info(f"Currently Pending: {current_pending}")
        logger.info(f"Currently in Sales: {current_submitted}")
        logger.info(f"Rejected (in period): {rejected_cnt}")
        logger.info(f"Returned (in period): {returned_cnt}")
        logger.info(f"Accepted (in period): {accepted_cnt}")
        
        # Calculate percentages based on uploads
        if uploads_cnt > 0:
            accepted_pct = round(100.0 * accepted_cnt / uploads_cnt, 1)
            rejected_pct = round(100.0 * (rejected_cnt + returned_cnt) / uploads_cnt, 1)
        else:
            accepted_pct = 0.0
            rejected_pct = 0.0
        
        # Completed vs Not Completed for pie chart (based on tasks, not versions)
        # A task is completed if it has any accepted version
        completed_tasks = 0
        for _, row in scope.iterrows():
            try:
                vh = json.loads(str(row.get('Version History') or '[]'))
                has_accepted = any(event.get('folder', '').lower() == 'accepted' for event in vh)
                if has_accepted:
                    completed_tasks += 1
            except:
                pass
        
        not_completed = max(total - completed_tasks, 0)
        completed = completed_tasks
        
        # Reviewer stats - calculate from videos that moved from pending to another status
        deltas = []
        reviewer_handled = 0  # Videos that were handled (moved from pending to rejected/submitted)
        reviewer_accepted = 0  # Videos that were accepted by reviewer (moved to submitted)
        
        # Debug logging
        logger.info(f"=== REVIEWER STATS CALCULATION ===")
        logger.info(f"Total tasks in scope: {len(scope)}")
        logger.info(f"Period mode: {mode}, Period value: {period}")
        
        # Track videos submitted to sales and their eventual outcomes
        reviewer_submitted_to_sales = 0
        reviewer_submitted_accepted = 0
        
        # Look at ALL tasks (not just scope) for complete reviewer analysis
        for task_idx, (_, row) in enumerate(df_all.iterrows()):
            try:
                task_num = row.get('Task #', 'NA')
                vh = json.loads(str(row.get('Version History') or '[]'))
                if not vh:
                    continue
                
                # Group events by version
                by_version = {}
                for event in vh:
                    v = event.get('version')
                    if v is None:
                        continue
                    if v not in by_version:
                        by_version[v] = []
                    by_version[v].append({
                        'folder': event.get('folder', '').lower(),
                        'at': event.get('at', ''),
                        'dt': None
                    })
                    # Try to parse datetime
                    try:
                        by_version[v][-1]['dt'] = datetime.strptime(event['at'], '%d-%m-%Y %H:%M:%S')
                    except Exception:
                        pass
                
                # Check each version for reviewer actions
                for v, events in by_version.items():
                    # Find pending event
                    pending_dt = None
                    submitted_to_sales = False
                    eventually_accepted = False
                    
                    for e in events:
                        if e['folder'] == 'pending' and e['dt']:
                            pending_dt = e['dt']
                        elif e['folder'] == 'submitted to sales':
                            submitted_to_sales = True
                        elif e['folder'] == 'accepted':
                            eventually_accepted = True
                    
                    # Calculate response times only for tasks in scope (filming date in period)
                    if pending_dt and row.get('Task #') in scope['Task #'].values:
                        # Check if reviewer handled it (moved to rejected or submitted)
                        for e in events:
                            if e['dt'] and e['dt'] > pending_dt:
                                if e['folder'] in ['submitted to sales', 'rejected']:
                                    reviewer_handled += 1
                                    if e['folder'] == 'submitted to sales':
                                        reviewer_accepted += 1
                                    # Calculate response time
                                    delta_hours = (e['dt'] - pending_dt).total_seconds() / 3600.0
                                    if delta_hours > 0:
                                        deltas.append(delta_hours)
                                    break
                    
                    # Track success rate: videos submitted to sales that eventually got accepted
                    if submitted_to_sales:
                        reviewer_submitted_to_sales += 1
                        if eventually_accepted:
                            reviewer_submitted_accepted += 1
                        
            except Exception as e:
                logger.error(f"Error processing reviewer stats for task {task_idx}: {e}")
        
        # Calculate reviewer success rate: % of videos submitted to sales that got accepted
        if reviewer_submitted_to_sales > 0:
            reviewer_success_rate = round(100.0 * reviewer_submitted_accepted / reviewer_submitted_to_sales, 1)
        else:
            reviewer_success_rate = 0.0
        
        logger.info(f"\n=== REVIEWER STATS SUMMARY ===")
        logger.info(f"Videos handled by reviewer (in period): {reviewer_handled}")
        logger.info(f"Videos sent to sales by reviewer (in period): {reviewer_accepted}")
        logger.info(f"Total videos submitted to sales (all time): {reviewer_submitted_to_sales}")
        logger.info(f"Videos eventually accepted after sales submission: {reviewer_submitted_accepted}")
        logger.info(f"Reviewer success rate (accepted/submitted to sales): {reviewer_success_rate}%")
        logger.info(f"Average response time: {sum(deltas)/len(deltas):.2f} hours" if deltas else "No data")
        
        # Calculate average response time safely
        avg_response = 0.0
        avg_response_display = "No data"
        if deltas and len(deltas) > 0:
            avg_response = sum(deltas) / len(deltas)
            # Ensure it's a valid number
            if not (0 <= avg_response < float('inf')):
                avg_response = 0.0
            else:
                # Format display based on duration
                if avg_response < 1:
                    # Less than 1 hour, show in minutes
                    avg_response_display = f"{round(avg_response * 60, 1)} minutes"
                else:
                    avg_response_display = f"{round(avg_response, 2)} hours"
        
        # For the API, still return hours as a number
        reviewer = {
            'avg_response_hours': round(avg_response, 2) if avg_response > 0 else 0.0,
            'avg_response_display': avg_response_display,  # Human-readable format
            'pending_videos': current_pending,  # Currently pending videos
            'handled_percent': reviewer_success_rate  # Success rate of handled videos
        }
        
        # Per-videographer mapping
        vmap = {}
        for _, row in scope.iterrows():
            videographer = row.get('Videographer') or 'Unassigned'
            task_number = int(row.get('Task #')) if pd.notna(row.get('Task #')) else None
            
            # Get timestamps from version history instead of timestamp columns
            last_pending = ''
            last_submitted = ''
            last_accepted = ''
            
            try:
                vh = json.loads(str(row.get('Version History') or '[]'))
                # Find the last occurrence of each event type
                for event in reversed(vh):  # Start from most recent
                    folder = event.get('folder', '').lower()
                    timestamp = event.get('at', '')
                    
                    if folder == 'pending' and not last_pending:
                        last_pending = timestamp
                    elif folder == 'submitted to sales' and not last_submitted:
                        last_submitted = timestamp
                    elif folder == 'accepted' and not last_accepted:
                        last_accepted = timestamp
                        
                    # Stop if we found all three
                    if last_pending and last_submitted and last_accepted:
                        break
            except:
                pass
            current_version = row.get('Current Version')
            if pd.isna(current_version) or current_version == '':
                current_version = None
            else:
                try:
                    current_version = int(current_version)
                except Exception:
                    pass
            versions = []
            try:
                vh = json.loads(row.get('Version History') or '[]')
                # Map folder names to proper status names
                folder_to_status = {
                    "raw": "Raw",
                    "pending": "Uploaded",  # Changed from Critique to Uploaded
                    "rejected": "Editing",
                    "submitted": "Submitted to Sales",
                    "accepted": "Done",
                    "returned": "Returned"
                }
                
                tmp = {}
                for ev in vh:
                    v = ev.get('version')
                    if v is None:
                        continue
                    folder = ev.get('folder', '').lower()
                    status = folder_to_status.get(folder, folder.title())
                    event_data = {"stage": status, "at": ev.get('at', '')}
                    
                    # Include rejection details if available
                    if 'rejection_class' in ev:
                        event_data['rejection_class'] = ev.get('rejection_class', 'Other')
                        event_data['rejection_comments'] = ev.get('rejection_comments', '')
                        event_data['rejected_by'] = ev.get('rejected_by', '')
                    
                    tmp.setdefault(v, []).append(event_data)
                versions = [{"version": v, "lifecycle": events} for v, events in sorted(tmp.items())]
            except Exception:
                versions = []
            item = {
                "task_number": task_number,
                "brand": safe_text(row.get('Brand')),
                "reference_number": safe_text(row.get('Reference Number')),
                "filming_date": safe_text(row.get('Filming Date')),
                "submitted_at": safe_text(last_pending),
                "current_version": current_version if current_version is not None else 'NA',
                "submitted_to_sales_at": safe_text(last_submitted),
                "accepted_at": safe_text(last_accepted),
                "versions": versions
            }
            vmap.setdefault(videographer, []).append(item)
        
        # Build per-videographer summary metrics from version history
        vg_names = sorted(list(set(scope['Videographer'].fillna('Unassigned').replace({'': 'Unassigned'}))))
        summary_videographers = {}
        
        logger.info(f"\n=== PER-VIDEOGRAPHER METRICS ===")
        logger.info(f"Videographers found: {vg_names}")
        
        for vg in vg_names:
            rows_vg = scope[(scope['Videographer'].fillna('Unassigned').replace({'': 'Unassigned'}) == vg)]
            logger.info(f"\n{vg}: {len(rows_vg)} tasks")
            
            # Count uploads and period events from version history
            vg_uploads = 0
            vg_rejected = 0
            vg_returned = 0
            vg_accepted = 0
            
            # Count current pending/submitted from ALL tasks (not just scope)
            vg_current_pending = 0
            vg_current_submitted = 0
            
            # Get ALL tasks for this videographer to check current status
            all_vg_tasks = df_all[(df_all['Videographer'].fillna('Unassigned').replace({'': 'Unassigned'}) == vg)]
            for _, row in all_vg_tasks.iterrows():
                status = str(row.get('Status', '')).strip()
                if status == 'Critique':
                    vg_current_pending += 1
                elif status == 'Submitted to Sales':
                    vg_current_submitted += 1
            
            # Count period events from scope tasks
            for _, row in rows_vg.iterrows():
                try:
                    task_num = row.get('Task #', 'NA')
                    vh = json.loads(str(row.get('Version History') or '[]'))
                    
                    # Track unique versions for uploads
                    unique_versions = set()
                    
                    # Count ALL events for tasks with filming date in period
                    for event in vh:
                        folder = event.get('folder', '').lower()
                        version = event.get('version', '?')
                        
                        if folder == 'pending':
                            unique_versions.add(version)
                        elif folder == 'rejected':
                            vg_rejected += 1
                        elif folder == 'returned':
                            vg_returned += 1
                        elif folder == 'accepted':
                            vg_accepted += 1
                    
                    # Add unique version count to uploads
                    vg_uploads += len(unique_versions)
                except:
                    pass
            
            logger.info(f"  - Uploads: {vg_uploads}")
            logger.info(f"  - Currently Pending: {vg_current_pending}")
            logger.info(f"  - Currently in Sales: {vg_current_submitted}")
            logger.info(f"  - Rejected: {vg_rejected}")
            logger.info(f"  - Accepted: {vg_accepted}")
            
            # Calculate acceptance percentage based on uploads
            if vg_uploads > 0:
                vg_accepted_pct = round(100.0 * vg_accepted / vg_uploads, 1)
            else:
                vg_accepted_pct = 0.0
            
            summary_videographers[vg] = {
                'total': int(rows_vg.shape[0]),
                'uploads': vg_uploads,
                'pending': vg_current_pending,
                'rejected': vg_rejected,
                'submitted_to_sales': vg_current_submitted,
                'returned': vg_returned,
                'accepted_videos': vg_accepted,
                'accepted_pct': vg_accepted_pct
            }
        
        summary = {
            "total": total,
            "assigned": assigned,
            "pending": current_pending,
            "rejected": rejected_cnt,
            "submitted_to_sales": current_submitted,
            "returned": returned_cnt,
            "uploads": uploads_cnt,
            "accepted_videos": accepted_cnt,
            "accepted_pct": accepted_pct,
            "rejected_pct": rejected_pct
        }
        
        return JSONResponse({
            "mode": mode,
            "period": period,
            "pie": {"completed": completed, "not_completed": not_completed},
            "summary": summary,
            "summary_videographers": summary_videographers,
            "reviewer": reviewer,
            "videographers": vmap
        })
    except Exception as e:
        logger.error(f"/api/dashboard error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# /api/stats endpoint removed - functionality merged into /api/dashboard

# Stats endpoint removed - functionality merged into main dashboard

# ========== STARTUP FUNCTIONS ==========
async def get_bot_user_id():
    """Get the bot's user ID from Slack"""
    global BOT_USER_ID
    try:
        response = await slack_client.auth_test()
        BOT_USER_ID = response["user_id"]
        logger.info(f"✅ Bot User ID retrieved: {BOT_USER_ID}")
        return BOT_USER_ID
    except Exception as e:
        logger.error(f"❌ Failed to get bot user ID: {e}")
        logger.error("Make sure SLACK_BOT_TOKEN is valid and has auth.test scope")
        return None

async def is_bot_mentioned(text: str) -> bool:
    """Check if the bot is mentioned in the message"""
    if not BOT_USER_ID:
        logger.warning("Bot user ID not set - cannot check mentions")
        return False
    
    # Check for direct @mention
    if f"<@{BOT_USER_ID}>" in text:
        return True
    
    # Also check for app_mention which might have different format
    # Sometimes Slack sends mentions as <!@USERID> for apps
    if f"<!@{BOT_USER_ID}>" in text:
        return True
    
    return False

async def get_channel_type(channel: str) -> str:
    """Get the type of channel (channel, group, im)"""
    try:
        response = await slack_client.conversations_info(channel=channel)
        channel_info = response["channel"]
        if channel_info.get("is_channel"):
            return "channel"
        elif channel_info.get("is_group"):
            return "group"
        elif channel_info.get("is_im"):
            return "im"
        elif channel_info.get("is_mpim"):
            return "mpim"  # Multi-party IM
        else:
            return "unknown"
    except Exception as e:
        logger.error(f"Failed to get channel type: {e}")
        return "unknown"

# ========== MAIN EXECUTION ==========
async def main():
    """Main async entry point"""
    # Check for required environment variables first
    if not SLACK_BOT_TOKEN or not SLACK_SIGNING_SECRET:
        logger.error("❌ Please set SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET environment variables")
        exit(1)
    
    if not OPENAI_API_KEY:
        logger.error("❌ Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    # Initialize Excel file
    await initialize_excel()
    
    # Get bot user ID - MUST happen before server starts
    bot_id = await get_bot_user_id()
    if not bot_id:
        logger.error("❌ Failed to retrieve bot user ID. Check your SLACK_BOT_TOKEN.")
        exit(1)
    
    # Run FastAPI server
    # Get port from environment variable for deployment
    port = int(os.getenv("PORT", 3000))
    
    config = uvicorn.Config(
        app=api,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    logger.info("🚀 FastAPI running on http://localhost:3000")
    logger.info("📚 API docs available at http://localhost:3000/docs")
    logger.info("🔗 Slack events endpoint: http://localhost:3000/slack/events")
    logger.info("🔗 Slack commands endpoint: http://localhost:3000/slack/slash-commands")
    
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())