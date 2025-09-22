"""
Video Upload and Approval System
Handles video uploads from videographers and approval workflow
"""

import asyncio
import fcntl
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import dropbox
import pandas as pd
import requests

from clients import slack_client
from config import CREDENTIALS_PATH, UAE_TZ, SLACK_BOT_TOKEN, OPENAI_API_KEY, VIDEOGRAPHER_CONFIG_PATH
from logger import logger
from simple_permissions import check_permission as simple_check_permission
from management import load_videographer_config

# Helper functions to get reviewer and head of sales info
def get_reviewer_info() -> Dict[str, str]:
    """Get reviewer info including user ID and channel ID"""
    config = load_videographer_config()
    reviewer = config.get("reviewer", {})
    return {
        "user_id": reviewer.get("slack_user_id", ""),
        "channel_id": reviewer.get("slack_channel_id", ""),
        "name": reviewer.get("name", "Reviewer"),
        "email": reviewer.get("email", "")
    }

def get_head_of_sales_info() -> Dict[str, str]:
    """Get head of sales info"""
    config = load_videographer_config()
    hos = config.get("head_of_sales", {})
    return {
        "user_id": hos.get("slack_user_id", ""),
        "channel_id": hos.get("slack_channel_id", ""),
        "name": hos.get("name", "Head of Sales"),
        "email": hos.get("email", "")
    }

# Dropbox folders
DROPBOX_FOLDERS = {
    "raw": "/Site Videos/Raw",
    "pending": "/Site Videos/Pending",
    "rejected": "/Site Videos/Rejected",
    "submitted": "/Site Videos/Submitted to Sales",
    "accepted": "/Site Videos/Accepted",
    "returned": "/Site Videos/Returned"
}

# Rejection categories with descriptions
REJECTION_CATEGORIES_WITH_DESCRIPTIONS = {
    "Previous Artwork is Visible": "When mocked up, the previous artwork is still visible from the sides, or the lights from it.",
    "Competitor Billboard Visible": "A competing advertiser's billboard is in the frame.",
    "Artwork Color is Incorrect": "The colour of the artwork appears different from the actual artwork itself & proof of play.",
    "Artwork Order is Incorrect": "The mocked up sequence of creatives plays in the wrong order, needs to be the same as proof of play.",
    "Environment Too Dark": "The scene lacks adequate lighting, causing the billboard and surroundings to appear dark. (Night)",
    "Environment Too Bright": "Excessive brightness or glare washes out the creative, reducing legibility. (Day)",
    "Blurry Artwork": "The billboard content appears out of focus in the video, impairing readability.",
    "Ghost Effect": "The cladding, when mocked up looks like cars going through it/lampposts when removed, can makes car disappear when passing through them.",
    "Cladding Lighting": "The external lighting on the billboard frame or cladding is dull/not accurate or off.",
    "Shaking Artwork or Cladding": "Structural vibration or instability results in a visibly shaky frame or creative playback.",
    "Shooting Angle": "The chosen camera angle distorts the artwork or makes it smaller (Billboard).",
    "Visible Strange Elements": "Unintended objects, example when mocked up cladding appearing in a different frame, artwork not going away on time etc.",
    "Transition Overlayer": "Video captures unintended transition animations or overlays, obscuring the main creative.",
    "Other": "Other reasons not covered by the above categories."
}

# Extract just the category names for the list
REJECTION_CATEGORIES = list(REJECTION_CATEGORIES_WITH_DESCRIPTIONS.keys())

# Import database functions for workflow persistence
from db_utils import (
    save_workflow_async, get_workflow_async, delete_workflow_async,
    get_all_pending_workflows_async
)

# Approval tracking - workflows persist in database
approval_workflows = {}  # workflow_id -> workflow_data (cache for active workflows)

async def get_workflow_with_cache(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Get workflow from cache or database"""
    # Check cache first
    if workflow_id in approval_workflows:
        return approval_workflows[workflow_id]
    
    # Load from database
    workflow = await get_workflow_async(workflow_id)
    if workflow:
        # Cache it
        approval_workflows[workflow_id] = workflow
    return workflow

async def classify_rejection_reason(comments: str) -> str:
    """Use OpenAI to classify rejection comments into predefined categories"""
    try:
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured, defaulting to 'Other'")
            return "Other"
            
        # Initialize OpenAI client
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Create the classification prompt with descriptions
        categories_list = "\n".join([
            f"- {category}: {description}"
            for category, description in REJECTION_CATEGORIES_WITH_DESCRIPTIONS.items()
        ])
        
        prompt = f"""Classify the following video rejection comment into EXACTLY one of these categories:
        
Categories:
{categories_list}

Comment: "{comments}"

Return ONLY the category name, nothing else."""
        
        # Call OpenAI API
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a video quality classifier. Respond only with the category name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50
        )
        
        # Extract the classification
        category = response.choices[0].message.content.strip()
        
        # Validate category
        if category in REJECTION_CATEGORIES:
            return category
        else:
            logger.warning(f"Invalid category returned by OpenAI: {category}, defaulting to 'Other'")
            return "Other"
            
    except Exception as e:
        logger.error(f"Error classifying rejection reason: {e}")
        return "Other"

async def get_rejection_history(task_number: int, include_current: bool = False) -> List[Dict[str, Any]]:
    """Get complete rejection/return history for a task from DB version history"""
    try:
        from db_utils import get_task_by_number as db_get
        row = db_get_task = await asyncio.to_thread(lambda: db_get(task_number))
        if not row:
            return []
        version_history_json = row.get('Version History', '[]') or '[]'
        version_history = json.loads(version_history_json)
        
        # Get all rejections and returns
        rejection_history = []
        for entry in version_history:
            folder = entry.get('folder', '').lower()
            
            # Include all rejected and returned entries
            if folder in ['rejected', 'returned']:
                rejected_by = entry.get('rejected_by', 'Unknown')
                
                # Determine the type based on folder and who rejected
                if folder == 'returned':
                    rejection_type = f"{rejected_by} Return"
                else:
                    rejection_type = f"{rejected_by} Rejection"
                
                rejection_history.append({
                    'version': entry.get('version', 1),
                    'class': entry.get('rejection_class', 'Other'),
                    'comments': entry.get('rejection_comments', ''),
                    'at': entry.get('at', ''),
                    'type': rejection_type,
                    'rejected_by': rejected_by
                })
                
        return rejection_history
        
    except Exception as e:
        logger.error(f"Error getting rejection history: {e}")
        return []

class DropboxManager:
    """Thread-safe Dropbox manager with file locking for multi-user support"""
    
    def __init__(self):
        self.dbx = None
        self._process_lock = asyncio.Lock()  # For async coordination within process
        self._last_refresh = 0
        self._token_lifetime = 3600  # 1 hour cache
        self._initialize()
    
    def _initialize(self):
        """Initialize with token refresh"""
        try:
            # Try to load existing valid token first
            if self._try_load_cached_token():
                return
            
            # Otherwise refresh
            self._refresh_token_with_lock()
        except Exception as e:
            logger.error(f"Failed to initialize Dropbox: {e}")
            raise
    
    def _try_load_cached_token(self) -> bool:
        """Try to load token if it's still valid"""
        try:
            with open(CREDENTIALS_PATH, "r") as f:
                creds = json.load(f)
                
            # Check if token was refreshed recently
            last_refresh = creds.get('last_refresh', 0)
            if time.time() - last_refresh < self._token_lifetime - 300:  # 5 min buffer
                self.dbx = dropbox.Dropbox(creds["access_token"])
                self._last_refresh = last_refresh
                logger.info("✅ Loaded cached Dropbox token")
                return True
        except Exception:
            pass
        return False
    
    def _refresh_token_with_lock(self):
        """Refresh token with file locking for multi-process safety"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(CREDENTIALS_PATH), exist_ok=True)
                
                # Open file for reading and writing
                with open(CREDENTIALS_PATH, "r+") as f:
                    # Acquire exclusive lock (blocks until available)
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    
                    try:
                        # Read current credentials
                        f.seek(0)
                        creds = json.load(f)
                        
                        # Double-check if another process just refreshed
                        last_refresh = creds.get('last_refresh', 0)
                        if time.time() - last_refresh < 60:  # Refreshed in last minute
                            self.dbx = dropbox.Dropbox(creds["access_token"])
                            self._last_refresh = last_refresh
                            logger.info("✅ Using recently refreshed token")
                            return
                        
                        # Make refresh request
                        logger.info("🔄 Refreshing Dropbox token...")
                        response = requests.post("https://api.dropbox.com/oauth2/token", data={
                            "grant_type": "refresh_token",
                            "refresh_token": creds["refresh_token"],
                            "client_id": creds["client_id"],
                            "client_secret": creds["client_secret"]
                        })
                        
                        if response.status_code == 200:
                            # Update credentials
                            new_token = response.json()["access_token"]
                            creds["access_token"] = new_token
                            creds["last_refresh"] = time.time()
                            
                            # Write back to file (atomic within lock)
                            f.seek(0)
                            json.dump(creds, f, indent=2)
                            f.truncate()  # Remove any leftover content
                            
                            # Update instance
                            self.dbx = dropbox.Dropbox(new_token)
                            self._last_refresh = creds["last_refresh"]
                            
                            logger.info("✅ Dropbox token refreshed successfully")
                            return
                        else:
                            raise Exception(f"Token refresh failed: {response.status_code} - {response.text}")
                            
                    finally:
                        # Lock is automatically released when file is closed
                        pass
                        
            except FileNotFoundError:
                logger.error(f"Credentials file not found: {CREDENTIALS_PATH}")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in credentials file: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                raise
            except Exception as e:
                logger.error(f"Token refresh error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise
    
    async def ensure_valid_token(self):
        """Ensure we have a valid token (async-safe)"""
        async with self._process_lock:
            # Check if token needs refresh
            if time.time() - self._last_refresh > self._token_lifetime - 300:
                await asyncio.to_thread(self._refresh_token_with_lock)
    
    async def get_shared_link(self, path: str) -> str:
        """Get a shareable link for a file in Dropbox"""
        # Ensure token is valid
        await self.ensure_valid_token()
        
        try:
            # First try to get existing shared links
            try:
                links = self.dbx.sharing_list_shared_links(path=path, direct_only=True)
                if links.links:
                    # Return the first existing link, modified for preview
                    url = links.links[0].url
                    if '?dl=1' in url:
                        url = url.replace('?dl=1', '?dl=0')
                    elif '?dl=0' not in url:
                        url = url + '?dl=0'
                    return url
            except Exception as e:
                logger.debug(f"No existing shared links found: {e}")
            
            # Create a new shared link if none exists
            try:
                # Create shared link with view access
                from dropbox.sharing import SharedLinkSettings
                # Create link without specifying visibility - will use default
                settings = SharedLinkSettings()
                shared_link_metadata = self.dbx.sharing_create_shared_link_with_settings(path, settings)
                # Modify the URL to force web preview instead of download
                url = shared_link_metadata.url
                if '?dl=1' in url:
                    url = url.replace('?dl=1', '?dl=0')
                elif '?dl=0' not in url:
                    url = url + '?dl=0'
                return url
            except Exception as e:
                if "shared_link_already_exists" in str(e):
                    # If link already exists, get it
                    links = self.dbx.sharing_list_shared_links(path=path, direct_only=True)
                    if links.links:
                        url = links.links[0].url
                        if '?dl=1' in url:
                            url = url.replace('?dl=1', '?dl=0')
                        elif '?dl=0' not in url:
                            url = url + '?dl=0'
                        return url
                else:
                    raise e
        except Exception as e:
            logger.error(f"Error creating shared link: {e}")
            # Fallback to temporary link
            try:
                result = self.dbx.files_get_temporary_link(path)
                return result.link
            except:
                return f"[Video uploaded to: {path}]"
    
    async def upload_video(self, file_content: bytes, filename: str, folder: str) -> str:
        """Upload video to Dropbox folder"""
        # Ensure token is valid
        await self.ensure_valid_token()
        
        try:
            # Ensure folder exists
            folder_path = DROPBOX_FOLDERS.get(folder)
            if not folder_path:
                raise ValueError(f"Invalid folder: {folder}")
            
            # Full path for the file
            full_path = f"{folder_path}/{filename}"
            
            # Upload with quality preservation settings
            # Using upload_session for large files to preserve quality
            file_size = len(file_content)
            CHUNK_SIZE = 4 * 1024 * 1024  # 4MB chunks
            
            if file_size <= CHUNK_SIZE:
                # Small file, single upload
                result = self.dbx.files_upload(
                    file_content,
                    full_path,
                    mode=dropbox.files.WriteMode.add,
                    autorename=True,
                    mute=True
                )
            else:
                # Large file, use upload session
                upload_session = self.dbx.files_upload_session_start(
                    file_content[:CHUNK_SIZE]
                )
                cursor = dropbox.files.UploadSessionCursor(
                    session_id=upload_session.session_id,
                    offset=CHUNK_SIZE
                )
                
                # Upload remaining chunks
                while cursor.offset < file_size:
                    chunk_end = min(cursor.offset + CHUNK_SIZE, file_size)
                    chunk = file_content[cursor.offset:chunk_end]
                    
                    if chunk_end < file_size:
                        self.dbx.files_upload_session_append_v2(chunk, cursor)
                        cursor.offset = chunk_end
                    else:
                        # Final chunk
                        result = self.dbx.files_upload_session_finish(
                            chunk,
                            cursor,
                            dropbox.files.CommitInfo(
                                path=full_path,
                                mode=dropbox.files.WriteMode.add,
                                autorename=True,
                                mute=True
                            )
                        )
                        cursor.offset = file_size
                        break
            
            logger.info(f"✅ Uploaded {filename} to {folder_path}")
            return result.path_display
            
        except Exception as e:
            logger.error(f"Error uploading to Dropbox: {e}")
            # Try refreshing token and retry once
            try:
                self._refresh_token()
                return await self.upload_video(file_content, filename, folder)
            except:
                raise
    
    async def move_file(self, from_path: str, to_folder: str) -> str:
        """Move file from one folder to another"""
        # Ensure token is valid
        await self.ensure_valid_token()
        
        try:
            # Get filename from path
            filename = os.path.basename(from_path)
            
            # Destination path
            to_folder_path = DROPBOX_FOLDERS.get(to_folder)
            if not to_folder_path:
                raise ValueError(f"Invalid folder: {to_folder}")
            
            to_path = f"{to_folder_path}/{filename}"
            
            # Move file
            result = self.dbx.files_move_v2(from_path, to_path)
            
            logger.info(f"✅ Moved {filename} to {to_folder_path}")
            return result.metadata.path_display
            
        except Exception as e:
            logger.error(f"Error moving file: {e}")
            raise

# Initialize Dropbox manager
dropbox_manager = DropboxManager()

async def check_existing_videos(task_number: int, reference_number: str) -> list:
    """Check for existing videos with same reference number in Dropbox"""
    existing_videos = []
    try:
        # Ensure token is valid before checking
        await dropbox_manager.ensure_valid_token()
        for folder_name, folder_path in DROPBOX_FOLDERS.items():
            try:
                # List files in folder
                result = dropbox_manager.dbx.files_list_folder(folder_path)
                
                # Check all files
                while True:
                    for entry in result.entries:
                        if isinstance(entry, dropbox.files.FileMetadata):
                            # Check if filename contains the reference number
                            if reference_number in entry.name:
                                existing_videos.append({
                                    'filename': entry.name,
                                    'folder': folder_name,
                                    'path': entry.path_display
                                })
                    
                    if not result.has_more:
                        break
                    result = dropbox_manager.dbx.files_list_folder_continue(result.cursor)
                    
            except Exception as e:
                logger.warning(f"Error checking folder {folder_path}: {e}")
    except Exception as e:
        logger.error(f"Error checking existing videos: {e}")
    
    return existing_videos

def extract_campaign_letter(filename: str) -> str:
    """Extract campaign letter (A, B, C...) from filename"""
    # Expected format: Ref_Location_Brand_MonthYY_Videographer_SalesPerson_Letter_Version
    parts = filename.rsplit('.', 1)[0].split('_')  # Remove extension and split
    if len(parts) >= 8:
        return parts[-2]  # Second to last is the letter
    return 'A'

def get_next_campaign_letter(existing_videos: list, reference_number: str) -> str:
    """Get the next available campaign letter based on existing videos"""
    if not existing_videos:
        return 'A'
    
    # Extract all used letters
    used_letters = set()
    for video in existing_videos:
        letter = extract_campaign_letter(video['filename'])
        if letter and letter.isalpha():
            used_letters.add(letter)
    
    # Find next available letter
    for i in range(26):  # A-Z
        letter = chr(65 + i)
        if letter not in used_letters:
            return letter
    
    return 'AA'  # Fallback if somehow all letters are used

async def handle_video_upload_by_task_number(channel: str, user_id: str, file_info: Dict[str, Any], task_number: int, destination_folder: str):
    """Handle video upload by task number - rename and version appropriately"""
    try:
        # Check permissions
        has_permission, error_msg = simple_check_permission(user_id, "upload_video")
        if not has_permission:
            await slack_client.chat_postMessage(
                channel=channel,
                text=error_msg
            )
            return False
        # Get task data first
        task_data = await get_task_data(task_number)
        if not task_data:
            await slack_client.chat_postMessage(
                channel=channel,
                text=f"❌ Task #{task_number} not found. Please check the task number."
            )
            return False
        
        # Check campaign end date
        campaign_end_date = task_data.get('Campaign End Date')
        if campaign_end_date:
            try:
                end_date = pd.to_datetime(campaign_end_date).date()
                today = datetime.now(UAE_TZ).date()
                
                if end_date < today:
                    await slack_client.chat_postMessage(
                        channel=channel,
                        text=f"❌ Cannot upload video: Campaign for Task #{task_number} ended on {end_date}. Please check with your supervisor."
                    )
                    return False
            except Exception as e:
                logger.warning(f"Error parsing campaign end date: {e}")
        
        # Get file details
        file_id = file_info.get("id")
        original_file_name = file_info.get("name", "video.mp4")
        
        # Get full file info from Slack
        file_response = await slack_client.files_info(file=file_id)
        if not file_response["ok"]:
            raise Exception(f"Failed to get file info: {file_response}")
        
        file_data = file_response["file"]
        file_url = file_data.get("url_private_download") or file_data.get("url_private")
        
        # Get user info
        user_info = await slack_client.users_info(user=user_id)
        user_name = user_info["user"]["profile"].get("real_name", "Unknown")
        
        # Download file from Slack
        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        response = requests.get(file_url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Read file content (preserving quality)
        file_content = response.content
        
        # Get reference number for version checking
        reference_number = str(task_data.get('Reference Number', '')).replace(' ', '')
        
        # Get latest version number (excluding accepted folder)
        latest_version = await get_latest_version_number(reference_number, exclude_accepted=True)
        new_version = latest_version + 1
        
        # Generate proper filename
        file_name = generate_video_filename(task_data, user_name, new_version)
        
        # Check for campaign letter (B, C, D...) if needed
        existing_videos = await check_existing_videos(task_number, reference_number)
        if existing_videos:
            # Update letter in filename
            next_letter = get_next_campaign_letter(existing_videos, reference_number)
            parts = file_name.rsplit('.', 1)[0].split('_')
            if len(parts) >= 8:
                parts[-2] = next_letter  # Update letter
                file_name = '_'.join(parts) + '.mp4'
        
        # Upload to Dropbox with generated filename
        dropbox_path = await dropbox_manager.upload_video(
            file_content,
            file_name,
            destination_folder
        )
        
        # Update Excel status with version only (no filename stored)
        await update_excel_status(task_number, destination_folder, version=new_version)
        
        # Send appropriate notifications
        if destination_folder == "pending":
            # Notify reviewer for approval
            await send_reviewer_approval(channel, file_name, dropbox_path, task_data, user_name)
            await slack_client.chat_postMessage(
                channel=channel,
                text=f"✅ Video for Task #{task_number} uploaded successfully as:\n`{file_name}`\n\nVersion: {new_version}\nStatus: Pending Review"
            )
        elif destination_folder == "raw":
            # Simple confirmation
            await slack_client.chat_postMessage(
                channel=channel,
                text=f"✅ Video for Task #{task_number} uploaded to Raw folder as:\n`{file_name}`\n\nVersion: {new_version}"
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Error handling video upload by task number: {e}")
        await slack_client.chat_postMessage(
            channel=channel,
            text=f"❌ Error uploading video: {str(e)}"
        )
        return False

async def handle_video_upload(channel: str, user_id: str, file_info: Dict[str, Any], destination_folder: str):
    """Handle video upload from Slack to Dropbox (legacy method)"""
    try:
        # Check permissions
        has_permission, error_msg = simple_check_permission(user_id, "upload_video")
        if not has_permission:
            await slack_client.chat_postMessage(
                channel=channel,
                text=error_msg
            )
            return False
        # Get file details
        file_id = file_info.get("id")
        file_name = file_info.get("name", "video.mp4")
        
        # Get full file info from Slack
        file_response = await slack_client.files_info(file=file_id)
        if not file_response["ok"]:
            raise Exception(f"Failed to get file info: {file_response}")
        
        file_data = file_response["file"]
        file_url = file_data.get("url_private_download") or file_data.get("url_private")
        
        # Get user info
        user_info = await slack_client.users_info(user=user_id)
        user_name = user_info["user"]["profile"].get("real_name", "Unknown")
        
        # Download file from Slack
        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        response = requests.get(file_url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Read file content (preserving quality)
        file_content = response.content
        
        # Extract task number from filename first to validate
        task_number = extract_task_number(file_name)
        
        # Get task details
        task_data = await get_task_data(task_number) if task_number else None
        
        if task_data:
            # Check campaign end date
            campaign_end_date = task_data.get('Campaign End Date')
            if campaign_end_date:
                try:
                    end_date = pd.to_datetime(campaign_end_date).date()
                    today = datetime.now(UAE_TZ).date()
                    
                    if end_date < today:
                        await slack_client.chat_postMessage(
                            channel=channel,
                            text=f"❌ Cannot upload video: Campaign end date ({end_date}) has already passed. Please check the campaign dates."
                        )
                        return False
                except Exception as e:
                    logger.warning(f"Error parsing campaign end date: {e}")
            
            # Check for duplicate videos
            reference_number = task_data.get('Reference Number', '').replace(' ', '')
            if reference_number:
                # Check for existing videos in both Dropbox and Excel
                existing_videos = await check_existing_videos(task_number, reference_number)
                existing_in_excel = await check_videos_in_excel(reference_number)
                
                # Combine both sources
                all_existing = existing_videos + existing_in_excel
                
                if all_existing:
                    # Generate appropriate filename with next letter
                    next_letter = get_next_campaign_letter(all_existing, reference_number)
                    
                    # Parse original filename and update with correct letter
                    base_name = file_name.rsplit('.', 1)[0]  # Remove extension
                    extension = file_name.rsplit('.', 1)[1] if '.' in file_name else 'mp4'
                    parts = base_name.split('_')
                    
                    # Update the letter in filename (assuming standard format)
                    if len(parts) >= 8:
                        parts[-2] = next_letter  # Update letter
                        updated_file_name = '_'.join(parts) + '.' + extension
                    else:
                        # If filename doesn't match expected format, append letter
                        updated_file_name = f"{base_name}_{next_letter}.{extension}"
                    
                    # Notify user about duplicate
                    await slack_client.chat_postMessage(
                        channel=channel,
                        text=f"ℹ️ Found {len(all_existing)} existing video(s) for reference {reference_number}. This will be video '{next_letter}'."
                    )
                    
                    # Update filename for upload
                    file_name = updated_file_name
        
        # Determine version number for legacy flow (use latest + 1 if task_data available)
        new_version: Optional[int] = None
        if task_data:
            reference_number = task_data.get('Reference Number', '').replace(' ', '')
            try:
                latest_version = await get_latest_version_number(reference_number, exclude_accepted=True)
                new_version = latest_version + 1
            except Exception:
                pass
        
        # Upload to Dropbox with final filename
        dropbox_path = await dropbox_manager.upload_video(
            file_content,
            file_name,
            destination_folder
        )
        
        # Update Excel status (if task number detected) with version only
        if task_number:
            await update_excel_status(task_number, destination_folder, version=new_version)
        
        # Send appropriate notifications based on folder
        if destination_folder == "pending":
            # Notify reviewer for approval
            await send_reviewer_approval(channel, file_name, dropbox_path, task_data, user_name)
        elif destination_folder == "raw":
            # Simple confirmation
            await slack_client.chat_postMessage(
                channel=channel,
                text=f"✅ Video uploaded to Raw folder successfully: `{file_name}`"
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Error handling video upload: {e}")
        await slack_client.chat_postMessage(
            channel=channel,
            text=f"❌ Error uploading video: {str(e)}"
        )
        return False

async def get_reviewer_channel() -> str:
    """Get the reviewer's Slack channel from config or env"""
    try:
        # Load config
        config_path = VIDEOGRAPHER_CONFIG_PATH
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        reviewer = config.get("reviewer", {})
        # Try slack_user_id first (for DMs), then channel_id
        if reviewer.get("slack_user_id"):
            return reviewer["slack_user_id"]
        elif reviewer.get("slack_channel_id"):
            return reviewer["slack_channel_id"]
        
        # Fallback to env var
        return os.getenv("REVIEWER_SLACK_CHANNEL", "")
    except Exception as e:
        logger.error(f"Error getting reviewer channel: {e}")
        return os.getenv("REVIEWER_SLACK_CHANNEL", "")

async def send_reviewer_approval(channel: str, filename: str, dropbox_path: str, task_data: Dict, uploader: str):
    """Send approval request to reviewer with interactive buttons"""
    try:
        # Get reviewer channel/user ID from config
        reviewer_channel = await get_reviewer_channel()
        if not reviewer_channel:
            reviewer_channel = channel  # Fallback to current channel
        
        # Extract task number from filename or task_data
        task_number = task_data.get('Task #', 0) if task_data else 0
        if not task_number:
            # Try to extract from filename
            task_match = re.search(r'Task(\d+)_', filename)
            if task_match:
                task_number = int(task_match.group(1))
        
        # Create workflow ID
        workflow_id = f"video_approval_{task_number}_{datetime.now().timestamp()}"
        
        # Get version from filename
        version_match = re.search(r'_(\d+)\.', filename)
        version = int(version_match.group(1)) if version_match else 1
        
        # Store workflow data
        workflow_data = {
            "task_number": task_number,
            "filename": filename,
            "dropbox_path": dropbox_path,
            "uploader": uploader,
            "upload_channel": channel,
            "task_data": task_data,
            "stage": "reviewer",
            "reviewer_id": reviewer_channel,
            "created_at": datetime.now(UAE_TZ).isoformat(),
            "version": version,
            "status": "pending"
        }
        
        # Save to database and cache
        approval_workflows[workflow_id] = workflow_data
        await save_workflow_async(workflow_id, workflow_data)
        
        # Create approval message with buttons
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"🎬 *New Video for Review*\n\n*File:* `{filename}`\n*Uploaded by:* {uploader}"
                }
            }
        ]
        
        # Add task details if available
        if task_data:
            blocks.append({
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Task #:* {task_data.get('Task #', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Brand:* {task_data.get('Brand', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Location:* {task_data.get('Location', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Sales Person:* {task_data.get('Sales Person', 'N/A')}"}
                ]
            })
        
        # Add video link
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"📹 <{await dropbox_manager.get_shared_link(dropbox_path)}|*Click to View/Download Video*>"
            }
        })
        
        # Add action buttons
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Accept"},
                    "style": "primary",
                    "action_id": "approve_video_workflow",
                    "value": workflow_id
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ Reject"},
                    "style": "danger",
                    "action_id": "reject_video_workflow",
                    "value": workflow_id
                }
            ]
        })
        
        # Send message
        result = await slack_client.chat_postMessage(
            channel=reviewer_channel,
            text=f"New video for review: {filename}",
            blocks=blocks
        )
        
        # Save message timestamp for recovery
        workflow_data['reviewer_msg_ts'] = result['ts']
        approval_workflows[workflow_id] = workflow_data
        await save_workflow_async(workflow_id, workflow_data)
        
    except Exception as e:
        logger.error(f"Error sending reviewer approval: {e}")
        raise

async def get_sales_person_channel(sales_person_name: str) -> str:
    """Get the Slack channel/user ID for a sales person"""
    try:
        # Load sales person config
        config_path = VIDEOGRAPHER_CONFIG_PATH
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if sales person has a Slack user ID configured
        sales_people = config.get("sales_people", {})
        if sales_person_name in sales_people:
            sales_info = sales_people[sales_person_name]
            if isinstance(sales_info, dict):
                # Try slack_user_id first (for DMs), then channel_id
                if sales_info.get("slack_user_id"):
                    return sales_info["slack_user_id"]
                elif sales_info.get("slack_channel_id"):
                    return sales_info["slack_channel_id"]
        
        # Try to find user by name in Slack as fallback
        try:
            response = await slack_client.users_list()
            if response["ok"]:
                for user in response["members"]:
                    if not user.get("is_bot") and not user.get("deleted"):
                        profile = user.get("profile", {})
                        real_name = profile.get("real_name", "")
                        display_name = profile.get("display_name", "")
                        
                        # Check if name matches
                        if (sales_person_name.lower() in real_name.lower() or 
                            sales_person_name.lower() in display_name.lower()):
                            # Update config with found user ID for future use
                            logger.info(f"Found Slack user ID for {sales_person_name}: {user['id']}")
                            return user["id"]
        except Exception as e:
            logger.warning(f"Error searching for user {sales_person_name}: {e}")
        
        # Fallback to general sales channel
        return os.getenv("SALES_SLACK_CHANNEL", "sales")
        
    except Exception as e:
        logger.error(f"Error getting sales person channel: {e}")
        return os.getenv("SALES_SLACK_CHANNEL", "sales")

async def send_sales_approval(filename: str, dropbox_path: str, task_data: Dict, sales_person: str):
    """Send approval request to sales person"""
    try:
        # Get sales person's Slack channel/user ID
        sales_channel = await get_sales_person_channel(sales_person)
        
        # Extract task number from filename or task_data
        task_number = task_data.get('Task #', 0) if task_data else 0
        if not task_number:
            # Try to extract from filename
            task_match = re.search(r'Task(\d+)_', filename)
            if task_match:
                task_number = int(task_match.group(1))
        
        # Create workflow ID
        workflow_id = f"sales_approval_{task_number}_{datetime.now().timestamp()}"
        
        # Get version from filename
        version_match = re.search(r'_(\d+)\.', filename)
        version = int(version_match.group(1)) if version_match else 1
        
        # Store workflow data
        workflow_data = {
            "task_number": task_number,
            "filename": filename,
            "dropbox_path": dropbox_path,
            "sales_person": sales_person,
            "task_data": task_data,
            "stage": "sales",
            "created_at": datetime.now(UAE_TZ).isoformat(),
            "version": version,
            "status": "pending"
        }
        
        # Save to database and cache
        approval_workflows[workflow_id] = workflow_data
        await save_workflow_async(workflow_id, workflow_data)
        
        # Create approval message
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"💼 *Video Submitted for Final Approval*\n\n*File:* `{filename}`\n*For:* {sales_person}"
                }
            }
        ]
        
        # Add task details
        if task_data:
            blocks.append({
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Task #:* {task_data.get('Task #', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Brand:* {task_data.get('Brand', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Location:* {task_data.get('Location', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Reference:* {task_data.get('Reference Number', 'N/A')}"}
                ]
            })
        
        # Add video link
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"📹 <{await dropbox_manager.get_shared_link(dropbox_path)}|*Click to View/Download Video*>"
            }
        })
        
        # Add action buttons
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Accept"},
                    "style": "primary",
                    "action_id": "approve_video_sales_workflow",
                    "value": workflow_id
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "↩️ Return for Revision"},
                    "style": "danger",
                    "action_id": "return_video_sales_workflow",
                    "value": workflow_id
                }
            ]
        })
        
        # Send message
        result = await slack_client.chat_postMessage(
            channel=sales_channel,
            text=f"Video submitted for approval: {filename}",
            blocks=blocks
        )
        
        # Save message timestamp for recovery
        workflow_data['sales_msg_ts'] = result['ts']
        approval_workflows[workflow_id] = workflow_data
        await save_workflow_async(workflow_id, workflow_data)
        
    except Exception as e:
        logger.error(f"Error sending sales approval: {e}")
        raise

async def handle_approval_action(action_data: Dict, user_id: str, response_url: str):
    """Handle approval/rejection button clicks"""
    try:
        # Parse action value
        value_data = json.loads(action_data["value"])
        filename = value_data["filename"]
        current_path = value_data["path"]
        action = value_data["action"]
        stage = value_data["stage"]
        task_data = value_data.get("task_data", {})
        
        # Determine destination folder based on action and stage
        if stage == "reviewer":
            if action == "accept":
                destination = "submitted"
                status_text = "✅ Video accepted and submitted to sales"
                next_stage = "sales"
            else:
                destination = "rejected"
                status_text = "❌ Video rejected and moved to rejected folder"
                next_stage = None
        else:  # sales stage
            if action == "accept":
                destination = "accepted"
                status_text = "✅ Video accepted and finalized"
                next_stage = None
            else:
                destination = "returned"
                status_text = "↩️ Video returned for revision"
                next_stage = None
        
        # Move file in Dropbox
        new_path = await dropbox_manager.move_file(current_path, destination)
        
        # Update Excel status
        task_number = task_data.get("Task #") or extract_task_number(filename)
        if task_number:
            await update_excel_status(task_number, destination)
            # Update Trello if needed
            await update_trello_status(task_number, destination)
        
        # Update the approval message
        await update_approval_message(response_url, filename, status_text, user_id)
        
        # Send next stage approval if needed
        if next_stage == "sales" and task_data:
            sales_person = task_data.get("Sales Person", "Sales Team")
            await send_sales_approval(filename, new_path, task_data, sales_person)
        
        # Notify original uploader if rejected/returned
        if action in ["reject", "return"] and stage == "reviewer":
            # Look for the workflow to get upload channel
            for workflow_id, workflow in approval_workflows.items():
                if workflow.get('filename') == filename and workflow.get('stage') == 'reviewer':
                    upload_channel = workflow.get('upload_channel')
                    if upload_channel:
                        await slack_client.chat_postMessage(
                            channel=upload_channel,
                            text=f"📹 Your video `{filename}` was rejected by the reviewer. Please check and resubmit."
                        )
                    break
        
    except Exception as e:
        logger.error(f"Error handling approval action: {e}")
        # Send error response
        await slack_client.chat_postMessage(
            channel=user_id,
            text=f"❌ Error processing approval: {str(e)}"
        )

async def update_approval_message(response_url: str, filename: str, status_text: str, user_id: str):
    """Update the approval message after action"""
    try:
        # Get user info
        user_info = await slack_client.users_info(user=user_id)
        user_name = user_info["user"]["profile"].get("real_name", "Unknown")
        
        # Update message
        requests.post(response_url, json={
            "replace_original": True,
            "text": f"{status_text}\n\nFile: `{filename}`\nActioned by: {user_name}"
        })
        
    except Exception as e:
        logger.error(f"Error updating approval message: {e}")

def extract_task_number(filename: str) -> Optional[int]:
    """Extract task number from video filename"""
    # Assuming filename contains task number like "Task123_..." or "123_..."
    import re
    
    # Try different patterns
    patterns = [
        r'Task[_\s]*(\d+)',  # Task123 or Task_123
        r'^(\d+)[_\s]',      # 123_ at start
        r'#(\d+)',           # #123
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return None

async def get_task_data(task_number: int) -> Optional[Dict]:
    """Get task data from Excel"""
    try:
        from db_utils import get_task as db_get_task
        task_data = await db_get_task(task_number)
        return task_data
    except Exception as e:
        logger.error(f"Error getting task data: {e}")
        return None

def generate_video_filename(task_data: Dict, videographer_name: str, version: int = 1) -> str:
    """Generate simplified video filename from task data"""
    try:
        task_number = task_data.get('Task #', 'Unknown')
        # Simple format: Task{number}_{version}.mp4
        filename = f"Task{task_number}_{version}.mp4"
        return filename
    except Exception as e:
        logger.error(f"Error generating filename: {e}")
        # Fallback filename
        return f"Task_Unknown_{version}.mp4"

async def get_latest_version_number(reference_number: str, exclude_accepted: bool = True) -> int:
    """Get the latest version number for a video across all folders"""
    latest_version = 0
    
    try:
        # Ensure token is valid before checking
        await dropbox_manager.ensure_valid_token()
        folders_to_check = DROPBOX_FOLDERS.copy()
        if exclude_accepted:
            # Remove accepted folder from search
            folders_to_check = {k: v for k, v in folders_to_check.items() if k != "accepted"}
        
        for folder_name, folder_path in folders_to_check.items():
            try:
                # List files in folder
                result = dropbox_manager.dbx.files_list_folder(folder_path)
                
                # Check all files
                while True:
                    for entry in result.entries:
                        if isinstance(entry, dropbox.files.FileMetadata):
                            # Check if filename contains the reference number
                            if reference_number in entry.name:
                                # Extract version number (last number before extension)
                                try:
                                    parts = entry.name.rsplit('.', 1)[0].split('_')
                                    if parts and parts[-1].isdigit():
                                        version = int(parts[-1])
                                        latest_version = max(latest_version, version)
                                except:
                                    continue
                    
                    if not result.has_more:
                        break
                    result = dropbox_manager.dbx.files_list_folder_continue(result.cursor)
                    
            except Exception as e:
                logger.warning(f"Error checking folder {folder_path}: {e}")
                
    except Exception as e:
        logger.error(f"Error getting latest version: {e}")
    
    return latest_version

async def update_excel_status(task_number: int, folder: str, version: Optional[int] = None, 
                             rejection_reason: Optional[str] = None, rejection_class: Optional[str] = None,
                             rejected_by: Optional[str] = None):
    """Update task status and related metadata in Excel based on folder; optionally record version and rejection info."""
    try:
        # Status mapping based on your workflow
        folder_to_status = {
            "Raw": "Raw",                      # Video uploaded to raw folder
            "Pending": "Critique",             # Video sent for review/critique
            "Rejected": "Editing",             # Video rejected, needs editing
            "Submitted to Sales": "Submitted to Sales", # Accepted by reviewer, sent to sales
            "Accepted": "Done",                # Accepted by sales, completed
            "Returned": "Returned",            # Returned by sales for revision
            # Keep lowercase mappings for backward compatibility
            "raw": "Raw",
            "pending": "Critique",
            "rejected": "Editing",
            "submitted": "Submitted to Sales",
            "accepted": "Done",
            "returned": "Returned",
            # Permanently Rejected videos also go to rejected folder
            "permanently_rejected": "Permanently Rejected"
        }
        
        new_status = folder_to_status.get(folder, "Unknown")
        
        # Update using DB
        from db_utils import get_task_by_number, update_status_with_history_and_timestamp
        task_row = get_task_by_number(task_number)
        if not task_row:
            logger.error(f"Task #{task_number} not found when updating status")
            return
        
        # Update status and history in DB
        success = update_status_with_history_and_timestamp(
            task_number=task_number,
            folder=folder,
            version=version,
            rejection_reason=rejection_reason,
            rejection_class=rejection_class,
            rejected_by=rejected_by
        )
        
        if not success:
            logger.error(f"Failed to update DB status for task {task_number}")
            return
        
        # If status is "Done", archive the task and remove from Excel
        if new_status == "Done":
            from db_utils import archive_task
            archive_task(task_number)
            logger.info(f"✅ Task #{task_number} moved to history DB and removed from Excel")
        else:
            logger.info(f"✅ Updated Task #{task_number} status to: {new_status}")
        
    except Exception as e:
        logger.error(f"Error updating Excel: {e}")

async def check_videos_in_excel(reference_number: str) -> list:
    """Deprecated Excel check; DB now prevents duplicates via unique reference."""
    return []

async def update_trello_status(task_number: int, folder: str):
    """Update Trello card based on video status"""
    try:
        # Import necessary modules
        import sys
        sys.path.append(str(Path(__file__).parent))
        from trello_utils import get_trello_card_by_task_number, set_trello_due_complete, archive_trello_card
        
        card = await asyncio.to_thread(get_trello_card_by_task_number, task_number)
        if not card:
            return
        
        # Update based on folder
        if folder == "submitted":
            await asyncio.to_thread(set_trello_due_complete, card['id'], True)
        elif folder == "returned":
            await asyncio.to_thread(set_trello_due_complete, card['id'], False)
        elif folder == "accepted":
            await asyncio.to_thread(archive_trello_card, card['id'])
        
    except Exception as e:
        logger.error(f"Error updating Trello: {e}")

# Videographer permissions
VIDEOGRAPHER_UPLOAD_PERMISSIONS = set()  # Will be populated from config

def load_videographer_permissions():
    """Load videographer permissions from config"""
    try:
        config_path = VIDEOGRAPHER_CONFIG_PATH
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Get all active videographers
        for name, info in config.get("videographers", {}).items():
            if info.get("active", True):
                VIDEOGRAPHER_UPLOAD_PERMISSIONS.add(name)
        
    except Exception as e:
        logger.error(f"Error loading videographer permissions: {e}")

# ========== NEW VIDEO UPLOAD WORKFLOW ==========

async def parse_task_number_from_message(message: str) -> Optional[int]:
    """Parse task number from user message"""
    import re
    
    try:
        # Try various patterns
        patterns = [
            r'task\s*#?\s*(\d+)',
            r'#(\d+)',
            r'\b(\d+)\b'  # Just numbers as fallback
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                return int(match.group(1))
                
    except Exception as e:
        logger.error(f"Error parsing task number: {e}")
    
    return None

async def handle_video_upload_with_parsing(channel: str, user_id: str, file_info: Dict[str, Any], message: str):
    """Handle video upload with task number parsing from message"""
    try:
        # Check permissions
        has_permission, error_msg = simple_check_permission(user_id, "upload_video")
        if not has_permission:
            await slack_client.chat_postMessage(
                channel=channel,
                text=error_msg
            )
            return
        # Parse task number
        task_number = await parse_task_number_from_message(message)
        if not task_number:
            await slack_client.chat_postMessage(
                channel=channel,
                text="❌ Please include the task number in your message (e.g., 'Task #5' or 'task 5')"
            )
            return
        
        # Get task data
        task_data = await get_task_data(task_number)
        if not task_data:
            await slack_client.chat_postMessage(
                channel=channel,
                text=f"❌ Task #{task_number} not found. Please check the task number."
            )
            return
        
        # Get task number for simplified naming
        task_number = task_data.get('Task #', 0)
        
        # Check existing versions to get next version number
        highest_version = 0
        base_filename = f"Task{task_number}"
        
        # Check all folders for existing files with this task number
        for folder_name, folder_path in DROPBOX_FOLDERS.items():
            if folder_name == "raw":  # Skip raw folder
                continue
                
            try:
                result = dropbox_manager.dbx.files_list_folder(folder_path)
                
                while True:
                    for entry in result.entries:
                        if hasattr(entry, 'name') and entry.name.startswith(base_filename + "_"):
                            # Extract version number from filename
                            # Format: Task{number}_{version}.extension
                            name_without_ext = entry.name.rsplit('.', 1)[0]
                            parts = name_without_ext.split('_')
                            if len(parts) == 2 and parts[-1].isdigit():
                                version = int(parts[-1])
                                highest_version = max(highest_version, version)
                    
                    if not result.has_more:
                        break
                    result = dropbox_manager.dbx.files_list_folder_continue(result.cursor)
                    
            except Exception as e:
                logger.warning(f"Error checking folder {folder_path}: {e}")
        
        # Next version
        new_version = highest_version + 1
        
        # Get file extension
        original_filename = file_info.get("name", "video.mp4")
        extension = original_filename.rsplit('.', 1)[-1] if '.' in original_filename else 'mp4'
        
        # Construct final filename with simplified format
        final_filename = f"Task{task_number}_{new_version}.{extension}"
        
        # Notify user
        await slack_client.chat_postMessage(
            channel=channel,
            text=f"📤 Uploading video for Task #{task_number}\nFilename: `{final_filename}`"
        )
        
        # Download file from Slack
        file_id = file_info.get("id")
        file_response = await slack_client.files_info(file=file_id)
        if not file_response["ok"]:
            raise Exception("Failed to get file info")
        
        file_data = file_response["file"]
        file_url = file_data.get("url_private_download") or file_data.get("url_private")
        
        # Download file
        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        import requests
        response = requests.get(file_url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to download file: {response.status_code}")
        
        file_content = response.content
        
        # Upload to Pending folder
        path = await dropbox_manager.upload_video(file_content, final_filename, "pending")
        
        # Update Excel status with the actual version number
        await update_excel_status(task_number, "Pending", version=new_version)
        
        # Send to reviewer for approval
        await send_video_to_reviewer(task_number, final_filename, path, user_id, task_data)
        
        await slack_client.chat_postMessage(
            channel=channel,
            text=f"✅ Video uploaded successfully!\n📁 Location: Pending folder\n🔍 Sent to reviewer for approval"
        )
        
    except Exception as e:
        logger.error(f"Error processing video upload: {e}")
        await slack_client.chat_postMessage(
            channel=channel,
            text=f"❌ Error uploading video: {str(e)}"
        )

async def send_video_to_reviewer(task_number: int, filename: str, dropbox_path: str, videographer_id: str, task_data: dict):
    """Send video to reviewer for approval"""
    try:
        # Get reviewer info
        from management import load_videographer_config
        config = load_videographer_config()
        reviewer = config.get("reviewer", {})
        reviewer_channel = reviewer.get("slack_channel_id")
        
        if not reviewer_channel:
            logger.error("Reviewer channel not configured")
            return
        
        # Create workflow ID
        workflow_id = f"video_approval_{task_number}_{datetime.now().timestamp()}"
        
        # Get version from filename
        version_match = re.search(r'_(\d+)\.', filename)
        version = int(version_match.group(1)) if version_match else 1
        
        # Store workflow data
        workflow_data = {
            "task_number": task_number,
            "filename": filename,
            "dropbox_path": dropbox_path,
            "videographer_id": videographer_id,
            "task_data": task_data,
            "stage": "reviewer",
            "created_at": datetime.now(UAE_TZ).isoformat(),
            "version": version,
            "status": "pending"
        }
        
        # Save to database and cache
        approval_workflows[workflow_id] = workflow_data
        await save_workflow_async(workflow_id, workflow_data)
        
        # Build message blocks
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*🎥 New Video Submission for Review*\n\n*Task #{task_number}* - `{filename}` (Version {version})"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Brand:* {task_data.get('Brand', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Location:* {task_data.get('Location', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Reference:* {task_data.get('Reference Number', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Sales Person:* {task_data.get('Sales Person', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Campaign Start:* {task_data.get('Campaign Start Date', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Campaign End:* {task_data.get('Campaign End Date', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Videographer:* {task_data.get('Videographer', 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Filming Date:* {task_data.get('Filming Date', 'N/A')}"},
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"📹 <{await dropbox_manager.get_shared_link(dropbox_path)}|*Click to View/Download Video*>"
                }
            }
        ]
        
        # Add version history if version > 1
        if version > 1:
            rejection_history = await get_rejection_history(task_number)
            if rejection_history:
                history_text = "*📋 Previous Rejections/Returns:*"
                for rejection in rejection_history:
                    rejection_type = rejection.get('type', 'Rejection')
                    timestamp = rejection.get('at', 'Unknown time')
                    history_text += f"\n• Version {rejection['version']} ({rejection_type}) - {rejection['class']}"
                    if rejection.get('comments'):
                        history_text += f": {rejection['comments']}"
                
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": history_text
                    }
                })
        
        # Add action buttons
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Accept"},
                    "style": "primary",
                    "action_id": "approve_video_reviewer",
                    "value": workflow_id
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ Reject"},
                    "style": "danger",
                    "action_id": "reject_video_reviewer",
                    "value": workflow_id
                }
            ]
        })
        
        # Send message with buttons
        result = await slack_client.chat_postMessage(
            channel=reviewer_channel,
            text=f"🎥 New video for review",
            blocks=blocks
        )
        
        # Save message timestamp for recovery
        workflow_data['reviewer_msg_ts'] = result['ts']
        approval_workflows[workflow_id] = workflow_data
        await save_workflow_async(workflow_id, workflow_data)
        
    except Exception as e:
        logger.error(f"Error sending to reviewer: {e}")

async def handle_reviewer_approval(workflow_id: str, user_id: str, response_url: str):
    """Handle reviewer approval - send directly to Head of Sales"""
    try:
        # Check permissions
        has_permission, error_msg = simple_check_permission(user_id, "approve_video_reviewer")
        if not has_permission:
            requests.post(response_url, json={
                "replace_original": True,
                "text": error_msg
            })
            return
        workflow = await get_workflow_with_cache(workflow_id)
        if not workflow:
            return
        
        task_number = workflow['task_number']
        filename = workflow['filename']
        task_data = workflow['task_data']
        videographer_id = workflow['videographer_id']
        
        # Move to Submitted to Sales
        from_path = workflow['dropbox_path']
        to_path = f"{DROPBOX_FOLDERS['submitted']}/{filename}"
        
        result = dropbox_manager.dbx.files_move_v2(from_path, to_path)
        workflow['dropbox_path'] = result.metadata.path_display
        
        # Get version from filename
        version_match = re.search(r'_(\d+)\.', filename)
        version = int(version_match.group(1)) if version_match else 1
        
        # Update status
        await update_excel_status(task_number, "Submitted to Sales", version=version)
        
        # Get video link
        video_link = await dropbox_manager.get_shared_link(to_path)
        
        # Notify videographer of reviewer approval with video link
        await slack_client.chat_postMessage(
            channel=videographer_id,
            text=f"✅ Good news! Your video for Task #{task_number} has been approved by the reviewer and sent to Head of Sales for final approval.\n\nFilename: `{filename}`\n\n📹 <{video_link}|Click to View Video>"
        )
        
        # Get Head of Sales info
        from management import load_videographer_config
        config = load_videographer_config()
        head_of_sales = config.get("head_of_sales", {})
        hos_channel = head_of_sales.get("slack_channel_id")
        
        if hos_channel:
            # Update workflow stage
            workflow['stage'] = 'hos'
            workflow['reviewer_approved'] = True
            workflow['reviewer_approved_by'] = user_id
            workflow['reviewer_approved_at'] = datetime.now(UAE_TZ).isoformat()
            
            # Save workflow to database
            await save_workflow_async(workflow_id, workflow)
            
            # Send to Head of Sales
            hos_result = await slack_client.chat_postMessage(
                channel=hos_channel,
                text=f"🎥 New video for final approval",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*🎥 Video for Final Approval*\n\n*Task #{task_number}* - `{filename}`"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Brand:* {task_data.get('Brand', 'N/A')}"},
                            {"type": "mrkdwn", "text": f"*Location:* {task_data.get('Location', 'N/A')}"},
                            {"type": "mrkdwn", "text": f"*Reference:* {task_data.get('Reference Number', 'N/A')}"},
                            {"type": "mrkdwn", "text": f"*Sales Person:* {task_data.get('Sales Person', 'N/A')}"},
                            {"type": "mrkdwn", "text": f"*Campaign Start:* {task_data.get('Campaign Start Date', 'N/A')}"},
                            {"type": "mrkdwn", "text": f"*Campaign End:* {task_data.get('Campaign End Date', 'N/A')}"},
                            {"type": "mrkdwn", "text": f"*Videographer:* {task_data.get('Videographer', 'N/A')}"},
                            {"type": "mrkdwn", "text": f"*Filming Date:* {task_data.get('Filming Date', 'N/A')}"},
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"📹 <{await dropbox_manager.get_shared_link(workflow['dropbox_path'])}|*Click to View/Download Video*>"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"_This video has been approved by the reviewer and is awaiting your final approval._"
                        }
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "✅ Accept"},
                                "style": "primary",
                                "action_id": "approve_video_hos",
                                "value": workflow_id
                            },
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "❌ Reject"},
                                "style": "danger",
                                "action_id": "reject_video_hos",
                                "value": workflow_id
                            }
                        ]
                    }
                ]
            )
            
            # Save HoS message timestamp
            workflow['hos_msg_ts'] = hos_result['ts']
            workflow['hos_id'] = hos_channel
            await save_workflow_async(workflow_id, workflow)
            
        else:
            logger.error("Head of Sales channel not configured")
        
        # Update reviewer's message with video link
        video_link = await dropbox_manager.get_shared_link(workflow['dropbox_path'])
        requests.post(response_url, json={
            "replace_original": True,
            "text": f"✅ Video accepted and sent to Head of Sales\nTask #{task_number}: `{filename}`",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"✅ *Video accepted and sent to Head of Sales*\n\nTask #{task_number}: `{filename}`"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"📹 <{video_link}|*Click to View/Download Video*>"
                    }
                }
            ]
        })
        
    except Exception as e:
        logger.error(f"Error handling reviewer approval: {e}")

async def handle_reviewer_rejection(workflow_id: str, user_id: str, response_url: str, rejection_comments: Optional[str] = None):
    """Handle reviewer rejection - send back to videographer"""
    try:
        # Check permissions
        has_permission, error_msg = simple_check_permission(user_id, "approve_video_reviewer")
        if not has_permission:
            requests.post(response_url, json={
                "replace_original": True,
                "text": error_msg
            })
            return
        workflow = await get_workflow_with_cache(workflow_id)
        if not workflow:
            return
        
        task_number = workflow['task_number']
        filename = workflow['filename']
        videographer_id = workflow['videographer_id']
        
        # Get current version from filename or workflow
        version_match = re.search(r'_(\d+)\.', filename)
        version = int(version_match.group(1)) if version_match else 1
        
        # Classify rejection reason if comments provided
        rejection_class = "Other"
        if rejection_comments:
            rejection_class = await classify_rejection_reason(rejection_comments)
        
        # Move to Rejected
        from_path = workflow['dropbox_path']
        to_path = f"{DROPBOX_FOLDERS['rejected']}/{filename}"
        
        result = dropbox_manager.dbx.files_move_v2(from_path, to_path)
        
        # Update status with rejection info
        await update_excel_status(task_number, "Rejected", version=version,
                                rejection_reason=rejection_comments,
                                rejection_class=rejection_class,
                                rejected_by="Reviewer")
        
        # Get video link from the new location (rejected folder)
        video_link = await dropbox_manager.get_shared_link(to_path)
        
        # Get rejection history
        rejection_history = await get_rejection_history(task_number)
        
        # Log rejection history for debugging
        logger.info(f"Task #{task_number} rejection history: {len(rejection_history)} entries")
        for idx, rej in enumerate(rejection_history):
            logger.info(f"  [{idx}] v{rej.get('version')} - {rej.get('type')} - {rej.get('class')}")
        
        # Create rejection history text including current rejection
        history_text = ""
        if rejection_history:
            history_text = "\n\n📋 *Rejection History:*"
            for rejection in rejection_history:
                rejection_type = rejection.get('type', 'Rejection')
                timestamp = rejection.get('at', 'Unknown time')
                history_text += f"\n• Version {rejection['version']} ({rejection_type}) at {timestamp}: {rejection['class']}"
                if rejection.get('comments'):
                    history_text += f" - {rejection['comments']}"
        
        # Notify videographer with formatted message
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"❌ *Your video for Task #{task_number} was rejected by the reviewer*\n\nFilename: `{filename}`\nVersion: {version}\n\n*Rejection Category:* {rejection_class}\n*Comments:* {rejection_comments or 'No comments provided'}\n\nPlease review the video and resubmit."
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"📹 <{video_link}|*Click to View Rejected Video*>"
                }
            }
        ]
        
        if history_text:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": history_text
                }
            })
        
        await slack_client.chat_postMessage(
            channel=videographer_id,
            text=f"❌ Video rejected by reviewer",
            blocks=blocks
        )
        
        # Update reviewer's message with rejection history
        reviewer_update_blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"❌ *Video rejected*\n\nTask #{task_number}: `{filename}`\n*Category:* {rejection_class}\n*Comments:* {rejection_comments or 'No comments'}\nVideographer has been notified."
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"📹 <{video_link}|*Click to View/Download Video*> (moved to Rejected folder)"
                }
            }
        ]
        
        # Add rejection history to reviewer's view
        if history_text:
            reviewer_update_blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": history_text
                }
            })
        
        requests.post(response_url, json={
            "replace_original": True,
            "text": f"❌ Video rejected\nTask #{task_number}: `{filename}`\nReason: {rejection_class}\nVideographer has been notified.",
            "blocks": reviewer_update_blocks
        })
        
        # Clean up workflow from cache and database
        if workflow_id in approval_workflows:
            del approval_workflows[workflow_id]
        await delete_workflow_async(workflow_id)
        
    except Exception as e:
        logger.error(f"Error handling reviewer rejection: {e}")

async def handle_sales_approval(workflow_id: str, user_id: str, response_url: str):
    """Handle sales approval - now sends to Head of Sales for final approval"""
    try:
        workflow = await get_workflow_with_cache(workflow_id)
        if not workflow:
            return
        
        task_number = workflow['task_number']
        filename = workflow['filename']
        videographer_id = workflow['videographer_id']
        task_data = workflow['task_data']
        
        # Mark sales approval in workflow
        workflow['sales_approved'] = True
        workflow['sales_approved_by'] = user_id
        workflow['sales_approved_at'] = datetime.now(UAE_TZ).isoformat()
        
        # Keep video in Submitted to Sales folder
        video_link = await dropbox_manager.get_shared_link(workflow['dropbox_path'])
        
        # Get Head of Sales info
        from management import load_videographer_config
        config = load_videographer_config()
        head_of_sales = config.get("head_of_sales", {})
        hos_channel = head_of_sales.get("slack_channel_id")
        
        # Get version from filename
        version_match = re.search(r'_(\d+)\.', filename)
        version = int(version_match.group(1)) if version_match else 1
        
        # Notify Head of Sales for final approval
        if hos_channel:
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"🔍 *Video Pending Head of Sales Approval*\n\n*Task #{task_number}* - `{filename}` (Version {version})\nBrand: {task_data.get('Brand', 'N/A')}\nLocation: {task_data.get('Location', 'N/A')}\n\n_Approved by Sales. Awaiting your final approval._"
                    }
                }
            ]
            
            # Add version history if version > 1
            if version > 1:
                rejection_history = await get_rejection_history(task_number)
                if rejection_history:
                    history_text = "*📋 Previous Rejections/Returns:*"
                    for rejection in rejection_history:
                        rejection_type = rejection.get('type', 'Rejection')
                        history_text += f"\n• Version {rejection['version']} ({rejection_type}) - {rejection['class']}"
                        if rejection.get('comments'):
                            history_text += f": {rejection['comments']}"
                    
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": history_text
                        }
                    })
            
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Video"},
                        "url": video_link,
                        "action_id": "view_video"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Accept"},
                        "style": "primary",
                        "value": workflow_id,
                        "action_id": "approve_video_hos"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌ Reject"},
                        "style": "danger",
                        "value": workflow_id,
                        "action_id": "reject_video_hos"
                    }
                ]
            })
            
            await slack_client.chat_postMessage(
                channel=hos_channel,
                text=f"Video for Task #{task_number} needs your approval",
                blocks=blocks
            )
        
        # Update original sales message
        requests.post(
            response_url,
            json={
                "text": "✅ Approved and sent to Head of Sales",
                "replace_original": True
            }
        )
        
        # Notify videographer of progress
        await slack_client.chat_postMessage(
            channel=videographer_id,
            text=f"✅ Good news! Your video for Task #{task_number} has been approved by sales and is now pending Head of Sales approval.\n\nFilename: `{filename}`"
        )
        
        # Notify reviewer of sales approval status
        reviewer = config.get("reviewer", {})
        reviewer_channel = reviewer.get("slack_channel_id")
        
        if reviewer_channel:
            await slack_client.chat_postMessage(
                channel=reviewer_channel,
                text=f"✅ Video approved by sales, pending Head of Sales approval\n\nTask #{task_number}: `{filename}`"
            )
        
    except Exception as e:
        logger.error(f"Error handling sales approval: {e}")

async def handle_sales_rejection(workflow_id: str, user_id: str, response_url: str, rejection_comments: Optional[str] = None):
    """Handle sales rejection - move to returned and notify"""
    try:
        workflow = await get_workflow_with_cache(workflow_id)
        if not workflow:
            return
        
        task_number = workflow['task_number']
        filename = workflow['filename']
        videographer_id = workflow['videographer_id']
        
        # Get current version from filename
        version_match = re.search(r'_(\d+)\.', filename)
        version = int(version_match.group(1)) if version_match else 1
        
        # Classify rejection reason if comments provided
        rejection_class = "Other"
        if rejection_comments:
            rejection_class = await classify_rejection_reason(rejection_comments)
        
        # Move to Returned
        from_path = workflow['dropbox_path']
        to_path = f"{DROPBOX_FOLDERS['returned']}/{filename}"
        
        result = dropbox_manager.dbx.files_move_v2(from_path, to_path)
        
        # Update status with rejection info
        await update_excel_status(task_number, "Returned", version=version,
                                rejection_reason=rejection_comments,
                                rejection_class=rejection_class,
                                rejected_by="Sales")
        
        # Also update returned timestamp since this is a sales rejection
        # update_movement_timestamp is deprecated, timestamps are updated via status update
        # Timestamp is now handled by update_excel_status above
        
        # Get the video link from the new location
        video_link = await dropbox_manager.get_shared_link(to_path)
        
        # Get rejection history (includes both rejected and returned)
        rejection_history = await get_rejection_history(task_number)
        
        # Create rejection history text including current rejection
        history_text = ""
        if rejection_history:
            history_text = "\n\n📋 *Rejection History:*"
            for rejection in rejection_history:
                rejection_type = rejection.get('type', 'Rejection')
                timestamp = rejection.get('at', 'Unknown time')
                history_text += f"\n• Version {rejection['version']} ({rejection_type}) at {timestamp}: {rejection['class']}"
                if rejection.get('comments'):
                    history_text += f" - {rejection['comments']}"
        
        # Get reviewer channel
        from management import load_videographer_config
        config = load_videographer_config()
        reviewer = config.get("reviewer", {})
        reviewer_channel = reviewer.get("slack_channel_id")
        
        # Notify videographer of sales rejection with formatted message
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"⚠️ *Your video for Task #{task_number} was rejected by sales after reviewer approval*\n\nFilename: `{filename}`\nVersion: {version}\n\n*Rejection Category:* {rejection_class}\n*Comments:* {rejection_comments or 'No comments provided'}\n\nThe video has been moved to the Returned folder. Please revise and resubmit."
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"📹 <{video_link}|*Click to View Returned Video*>"
                }
            }
        ]
        
        if history_text:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": history_text
                }
            })
        
        await slack_client.chat_postMessage(
            channel=videographer_id,
            text=f"⚠️ Video rejected by sales",
            blocks=blocks
        )
        
        # Notify reviewer with formatted message and video link including history
        if reviewer_channel:
            reviewer_blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"⚠️ *Video rejected by sales*\n\nTask #{task_number}: `{filename}`\n*Category:* {rejection_class}\n*Comments:* {rejection_comments or 'No comments'}\nThe video has been moved to the Returned folder."
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"📹 <{video_link}|*Click to View/Download Video*> (moved to Returned folder)"
                    }
                }
            ]
            
            # Add rejection history
            if history_text:
                reviewer_blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": history_text
                    }
                })
            
            await slack_client.chat_postMessage(
                channel=reviewer_channel,
                text=f"⚠️ Video rejected by sales",
                blocks=reviewer_blocks
            )
        
        # Update sales message
        requests.post(response_url, json={
            "replace_original": True,
            "text": (f"❌ Video rejected and returned\n"
                    f"Task #{task_number}: `{filename}`\n"
                    f"Category: {rejection_class}\n"
                    f"Videographer and reviewer have been notified."),
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"❌ *Video rejected and returned*\n\nTask #{task_number}: `{filename}`\n*Category:* {rejection_class}\n*Comments:* {rejection_comments or 'No comments'}\nVideographer and reviewer have been notified."
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"📹 <{video_link}|*Click to View/Download Video*> (moved to Returned folder)"
                    }
                }
            ]
        })
        
        # Clean up workflow from cache and database
        if workflow_id in approval_workflows:
            del approval_workflows[workflow_id]
        await delete_workflow_async(workflow_id)
        
    except Exception as e:
        logger.error(f"Error handling sales rejection: {e}")

async def update_task_status(task_number: int, status: str):
    """Update task status in DB and stamp movement timestamp"""
    try:
        from db_utils import update_task_by_number as db_update
        ok = await asyncio.to_thread(lambda: db_update(task_number, {'Status': status}))
        if ok:
            logger.info(f"Updated Task #{task_number} status to: {status}")
        else:
            logger.error(f"DB update failed for Task #{task_number}")
    except Exception as e:
        logger.error(f"Error updating task status: {e}")

async def archive_and_remove_completed_task(task_number: int):
    """Archive completed task to historical DB and archive Trello card"""
    try:
        from db_utils import archive_task
        ok = archive_task(task_number)
        if ok:
            logger.info(f"✅ Task #{task_number} moved to history DB")
            from trello_utils import get_trello_card_by_task_number, archive_trello_card
            card = await get_trello_card_by_task_number(task_number)
            if card:
                await asyncio.to_thread(archive_trello_card, card['id'])
                logger.info(f"Archived Trello card for Task #{task_number}")
            else:
                logger.warning(f"No Trello card found for Task #{task_number}")
    except Exception as e:
        logger.error(f"Error archiving completed task: {e}")

async def handle_hos_approval(workflow_id: str, user_id: str, response_url: str):
    """Handle Head of Sales approval - final acceptance"""
    try:
        # Check permissions
        has_permission, error_msg = simple_check_permission(user_id, "approve_video_hos")
        if not has_permission:
            requests.post(response_url, json={
                "replace_original": True,
                "text": error_msg
            })
            return
        workflow = await get_workflow_with_cache(workflow_id)
        if not workflow:
            return
        
        # Verify reviewer approval was given first
        if not workflow.get('reviewer_approved'):
            logger.error(f"HoS approval attempted without reviewer approval for workflow {workflow_id}")
            return
        
        task_number = workflow['task_number']
        filename = workflow['filename']
        videographer_id = workflow['videographer_id']
        task_data = workflow['task_data']
        
        # Now move to Accepted folder
        from_path = workflow['dropbox_path']
        to_path = f"{DROPBOX_FOLDERS['accepted']}/{filename}"
        
        result = dropbox_manager.dbx.files_move_v2(from_path, to_path)
        
        # Get version from filename
        version_match = re.search(r'_(\d+)\.', filename)
        version = int(version_match.group(1)) if version_match else 1
        
        # Update status to Done
        await update_excel_status(task_number, "Accepted", version=version)
        
        # CLEANUP: Find and reject all other versions of this task
        await cleanup_other_versions(task_number, version)
        
        # Archive task to historical DB and remove from Trello
        await archive_and_remove_completed_task(task_number)
        
        # Get the video link from accepted folder
        video_link = await dropbox_manager.get_shared_link(to_path)
        
        # Notify videographer of final acceptance with video link
        await slack_client.chat_postMessage(
            channel=videographer_id,
            text=f"🎉 Excellent news! Your video for Task #{task_number} has been fully accepted by Head of Sales!\n\nFilename: `{filename}`\nStatus: Done\n\n📹 <{video_link}|Click to View Final Video>"
        )
        
        # Load config for notifications
        from management import load_videographer_config
        config = load_videographer_config()
        
        # Notify reviewer of final acceptance
        reviewer = config.get("reviewer", {})
        reviewer_channel = reviewer.get("slack_channel_id")
        
        if reviewer_channel:
            await slack_client.chat_postMessage(
                channel=reviewer_channel,
                text=f"✅ Video fully accepted by Head of Sales\n\nTask #{task_number}: `{filename}`\nThe video has been moved to the Accepted folder.\n\n📹 <{video_link}|Click to View Final Video>"
            )
        
        # Notify sales person with final link - they can now use it
        sales_person_name = task_data.get('Sales Person', '')
        sales_people = config.get("sales_people", {})
        sales_person = sales_people.get(sales_person_name, {})
        sales_channel = sales_person.get("slack_channel_id")
        
        if sales_channel:
            await slack_client.chat_postMessage(
                channel=sales_channel,
                text=f"🎉 Video ready for use - Approved by Head of Sales",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"🎉 *Video Ready for Use*\n\nTask #{task_number}\nFilename: `{filename}`\nBrand: {task_data.get('Brand', '')}\nLocation: {task_data.get('Location', '')}\nCampaign: {task_data.get('Campaign Start Date', '')} to {task_data.get('Campaign End Date', '')}\n\n_This video has been approved by both the Reviewer and Head of Sales and is ready for your campaign._"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"📹 <{video_link}|*Download Final Video*>"
                        }
                    }
                ]
            )
        
        # Update HoS message
        requests.post(response_url, json={
            "replace_original": True,
            "text": f"✅ Video accepted and all parties notified\nTask #{task_number}: `{filename}`"
        })
        
        # Clean up workflow from cache and database
        if workflow_id in approval_workflows:
            del approval_workflows[workflow_id]
        await delete_workflow_async(workflow_id)
        
    except Exception as e:
        logger.error(f"Error handling HoS approval: {e}")

async def handle_hos_rejection(workflow_id: str, user_id: str, response_url: str, rejection_comments: Optional[str] = None):
    """Handle Head of Sales rejection - move back to returned"""
    try:
        # Check permissions
        has_permission, error_msg = simple_check_permission(user_id, "approve_video_hos")
        if not has_permission:
            requests.post(response_url, json={
                "replace_original": True,
                "text": error_msg
            })
            return
        workflow = await get_workflow_with_cache(workflow_id)
        if not workflow:
            return
        
        task_number = workflow['task_number']
        filename = workflow['filename']
        videographer_id = workflow['videographer_id']
        
        # Get current version from filename
        version_match = re.search(r'_(\d+)\.', filename)
        version = int(version_match.group(1)) if version_match else 1
        
        # Classify rejection reason if comments provided
        rejection_class = "Other"
        if rejection_comments:
            rejection_class = await classify_rejection_reason(rejection_comments)
        
        # Move to Returned folder
        from_path = workflow['dropbox_path']
        to_path = f"{DROPBOX_FOLDERS['returned']}/{filename}"
        
        result = dropbox_manager.dbx.files_move_v2(from_path, to_path)
        
        # Update status with rejection info
        await update_excel_status(task_number, "Returned", version=version,
                                rejection_reason=rejection_comments,
                                rejection_class=rejection_class,
                                rejected_by="Head of Sales")
        
        # Update returned timestamp
        # update_movement_timestamp is deprecated, timestamps are updated via status update
        # Timestamp is now handled by update_excel_status above
        
        # Get the video link from returned folder
        video_link = await dropbox_manager.get_shared_link(to_path)
        
        # Get rejection history
        rejection_history = await get_rejection_history(task_number)
        
        # Log rejection history for debugging
        logger.info(f"Task #{task_number} rejection history: {len(rejection_history)} entries")
        for idx, rej in enumerate(rejection_history):
            logger.info(f"  [{idx}] v{rej.get('version')} - {rej.get('type')} - {rej.get('class')}")
        
        # Create rejection history text including current rejection
        history_text = ""
        if rejection_history:
            history_text = "\n\n📋 *Rejection History:*"
            for rejection in rejection_history:
                rejection_type = rejection.get('type', 'Rejection')
                timestamp = rejection.get('at', 'Unknown time')
                history_text += f"\n• Version {rejection['version']} ({rejection_type}) at {timestamp}: {rejection['class']}"
                if rejection.get('comments'):
                    history_text += f" - {rejection['comments']}"
        
        # Load config for notifications
        from management import load_videographer_config
        config = load_videographer_config()
        
        # Notify videographer of HoS rejection
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"⚠️ *Your video for Task #{task_number} was rejected by Head of Sales*\n\nFilename: `{filename}`\nVersion: {version}\n\n*Rejection Category:* {rejection_class}\n*Comments:* {rejection_comments or 'No comments provided'}\n\n_Note: This video was previously approved by both reviewer and sales_\n\nThe video has been moved to the Returned folder. Please revise and resubmit."
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"📹 <{video_link}|*Click to View Returned Video*>"
                }
            }
        ]
        
        if history_text:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": history_text
                }
            })
        
        await slack_client.chat_postMessage(
            channel=videographer_id,
            text=f"⚠️ Video rejected by Head of Sales",
            blocks=blocks
        )
        
        # Notify reviewer with rejection history
        reviewer = config.get("reviewer", {})
        reviewer_channel = reviewer.get("slack_channel_id")
        
        if reviewer_channel:
            reviewer_blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"⚠️ *Video rejected by Head of Sales*\n\nTask #{task_number}: `{filename}`\n*Category:* {rejection_class}\n*Comments:* {rejection_comments or 'No comments'}\nStatus: Editing (Returned)\n\nThe video has been moved back to the Returned folder."
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"📹 <{video_link}|*Click to View/Download Video*> (moved to Returned folder)"
                    }
                }
            ]
            
            # Add rejection history
            if history_text:
                reviewer_blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": history_text
                    }
                })
            
            await slack_client.chat_postMessage(
                channel=reviewer_channel,
                text=f"⚠️ Video rejected by Head of Sales",
                blocks=reviewer_blocks
            )
        
        
        # Update HoS message
        requests.post(response_url, json={
            "replace_original": True,
            "text": f"❌ Video rejected and returned\nTask #{task_number}: `{filename}`\n*Category:* {rejection_class}\nAll parties have been notified."
        })
        
        # Clean up workflow from cache and database
        if workflow_id in approval_workflows:
            del approval_workflows[workflow_id]
        await delete_workflow_async(workflow_id)
        
    except Exception as e:
        logger.error(f"Error handling HoS rejection: {e}")


async def cleanup_other_versions(task_number: int, accepted_version: int):
    """When a video is accepted, find and reject all other versions"""
    try:
        logger.info(f"Starting cleanup for Task #{task_number}, accepted version: {accepted_version}")
        
        # Search pattern for this task's videos
        base_filename = f"Task{task_number}_"
        videos_to_reject = []
        workflows_to_cancel = []
        
        # Search only folders in active pipeline (not rejected/returned)
        folders_to_search = ["pending", "submitted"]
        
        for folder_name in folders_to_search:
            folder_path = DROPBOX_FOLDERS.get(folder_name)
            if not folder_path:
                continue
                
            try:
                # List files in folder
                result = dropbox_manager.dbx.files_list_folder(folder_path)
                
                while True:
                    for entry in result.entries:
                        if isinstance(entry, dropbox.files.FileMetadata):
                            # Check if this is a video for the same task
                            if entry.name.startswith(base_filename):
                                # Extract version from filename
                                version_match = re.search(r'_(\d+)\.', entry.name)
                                if version_match:
                                    file_version = int(version_match.group(1))
                                    # Don't reject the accepted version
                                    if file_version != accepted_version:
                                        videos_to_reject.append({
                                            'path': entry.path_display,
                                            'filename': entry.name,
                                            'folder': folder_name,
                                            'version': file_version
                                        })
                    
                    # Check if there are more files
                    if not result.has_more:
                        break
                    result = dropbox_manager.dbx.files_list_folder_continue(result.cursor)
                    
            except Exception as e:
                logger.error(f"Error searching folder {folder_name}: {e}")
        
        logger.info(f"Found {len(videos_to_reject)} videos to reject for Task #{task_number}")
        
        # Move all found videos to rejected folder
        for video in videos_to_reject:
            try:
                # Move to rejected folder
                to_path = f"{DROPBOX_FOLDERS['rejected']}/{video['filename']}"
                
                # Check if file already exists in rejected folder
                try:
                    dropbox_manager.dbx.files_get_metadata(to_path)
                    # File exists, need to rename
                    base_name = video['filename'].rsplit('.', 1)[0]
                    extension = video['filename'].rsplit('.', 1)[1] if '.' in video['filename'] else 'mp4'
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    to_path = f"{DROPBOX_FOLDERS['rejected']}/{base_name}_dup_{timestamp}.{extension}"
                except:
                    # File doesn't exist, proceed normally
                    pass
                
                dropbox_manager.dbx.files_move_v2(video['path'], to_path)
                
                logger.info(f"Moved {video['filename']} from {video['folder']} to rejected")
                
                # Update database status to Rejected
                # Note: We skip this update because the task is already marked as Done/accepted
                # and may have been archived. The rejection is only recorded in the file move.
                logger.info(f"Video {video['filename']} v{video['version']} auto-rejected (version {accepted_version} was accepted)")
                
                # Find and cancel any active workflows for this video
                for workflow_id, workflow in list(approval_workflows.items()):
                    if (workflow.get('task_number') == task_number and 
                        workflow.get('filename') == video['filename']):
                        workflows_to_cancel.append((workflow_id, workflow))
                
            except Exception as e:
                logger.error(f"Error moving {video['filename']} to rejected: {e}")
        
        # Also check database for workflows not in memory
        try:
            all_pending_workflows = await get_all_pending_workflows_async()
            for workflow in all_pending_workflows:
                if (workflow.get('task_number') == task_number and 
                    workflow.get('workflow_id') not in [w[0] for w in workflows_to_cancel]):
                    # Extract version from workflow filename
                    version_match = re.search(r'_(\d+)\.', workflow.get('filename', ''))
                    if version_match:
                        file_version = int(version_match.group(1))
                        if file_version != accepted_version:
                            workflows_to_cancel.append((workflow['workflow_id'], workflow))
        except Exception as e:
            logger.error(f"Error checking database for workflows: {e}")
        
        # Cancel active workflows and update messages
        for workflow_id, workflow in workflows_to_cancel:
            try:
                logger.info(f"Canceling workflow {workflow_id} for {workflow['filename']}")
                
                # Update Slack messages based on workflow stage
                stage = workflow.get('stage', '')
                
                if stage == 'reviewer' and workflow.get('reviewer_msg_ts'):
                    # Update reviewer message
                    reviewer_channel = (await get_reviewer_channel()) or workflow.get('reviewer_id')
                    if reviewer_channel and workflow.get('reviewer_msg_ts'):
                        try:
                            await slack_client.chat_update(
                                channel=reviewer_channel,
                                ts=workflow['reviewer_msg_ts'],
                                text=f"⚠️ This approval request is no longer valid - another version was accepted",
                                blocks=[{
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": f"⚠️ *This approval request is no longer valid*\n\nTask #{task_number}: `{workflow['filename']}`\n\nAnother version (v{accepted_version}) has been accepted. This video has been moved to rejected."
                                    }
                                }]
                            )
                        except Exception as e:
                            logger.warning(f"Could not update reviewer message for {workflow['filename']}: {e}")
                    else:
                        logger.debug(f"No reviewer message to update for workflow {workflow_id}")
                
                elif stage == 'hos' and workflow.get('hos_msg_ts'):
                    # Update HoS message
                    hos_channel = workflow.get('hos_id')
                    if hos_channel:
                        try:
                            await slack_client.chat_update(
                                channel=hos_channel,
                                ts=workflow['hos_msg_ts'],
                                text=f"⚠️ This approval request is no longer valid - another version was accepted",
                                blocks=[{
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": f"⚠️ *This approval request is no longer valid*\n\nTask #{task_number}: `{workflow['filename']}`\n\nAnother version (v{accepted_version}) has been accepted. This video has been moved to rejected."
                                    }
                                }]
                            )
                        except Exception as e:
                            logger.warning(f"Could not update HoS message: {e}")
                
                # Remove workflow from cache and database
                if workflow_id in approval_workflows:
                    del approval_workflows[workflow_id]
                await delete_workflow_async(workflow_id)
                
            except Exception as e:
                logger.error(f"Error canceling workflow {workflow_id}: {e}")
        
        # Notify videographers about auto-rejected videos
        videographers_notified = set()
        for video in videos_to_reject:
            try:
                # Find videographer for this video
                videographer_id = None
                for workflow_id, workflow in workflows_to_cancel:
                    if workflow.get('filename') == video['filename']:
                        videographer_id = workflow.get('videographer_id')
                        break
                
                if videographer_id and videographer_id not in videographers_notified:
                    videographers_notified.add(videographer_id)
                    
                    # Count how many videos were rejected for this videographer
                    videographer_videos = [v for v in videos_to_reject if any(
                        w[1].get('videographer_id') == videographer_id and w[1].get('filename') == v['filename'] 
                        for w in workflows_to_cancel
                    )]
                    
                    if videographer_videos:
                        await slack_client.chat_postMessage(
                            channel=videographer_id,
                            text=f"ℹ️ Your videos for Task #{task_number} have been auto-rejected",
                            blocks=[{
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"ℹ️ *Videos Auto-Rejected*\n\nTask #{task_number} - Version {accepted_version} has been accepted.\n\nThe following videos in the approval pipeline were automatically moved to rejected:\n" + 
                                           "\n".join([f"• `{v['filename']}` (v{v['version']}) from {v['folder']}" for v in videographer_videos])
                                }
                            }]
                        )
            except Exception as e:
                logger.error(f"Error notifying videographer: {e}")
        
        logger.info(f"Cleanup completed for Task #{task_number}")
        
    except Exception as e:
        logger.error(f"Error in cleanup_other_versions: {e}")


async def handle_permanent_rejection(task_number: int, task_data: Dict[str, Any]):
    """Handle permanent rejection of a task - reject all videos and cancel workflows"""
    try:
        logger.info(f"Processing permanent rejection for Task #{task_number}")
        
        # Get videographer info
        videographer_name = task_data.get('Videographer', '')
        videographer_id = None
        
        # Load config to get videographer Slack ID
        from management import load_videographer_config
        config = load_videographer_config()
        videographers = config.get('videographers', {})
        
        if videographer_name and videographer_name in videographers:
            videographer_info = videographers[videographer_name]
            videographer_id = videographer_info.get('slack_user_id') if isinstance(videographer_info, dict) else None
        
        # Search for all videos of this task
        base_filename = f"Task{task_number}_"
        videos_rejected = []
        
        # Search all folders for this task's videos
        all_folders = ["raw", "pending", "submitted", "returned", "rejected"]
        
        for folder_name in all_folders:
            folder_path = DROPBOX_FOLDERS.get(folder_name)
            if not folder_path:
                continue
                
            try:
                # List files in folder
                result = dropbox_manager.dbx.files_list_folder(folder_path)
                
                while True:
                    for entry in result.entries:
                        if isinstance(entry, dropbox.files.FileMetadata):
                            # Check if this is a video for the task
                            if entry.name.startswith(base_filename):
                                # If not already in rejected folder, move it
                                if folder_name != "rejected":
                                    try:
                                        # Move to rejected folder
                                        to_path = f"{DROPBOX_FOLDERS['rejected']}/{entry.name}"
                                        
                                        # Check if file already exists in rejected folder
                                        try:
                                            dropbox_manager.dbx.files_get_metadata(to_path)
                                            # File exists, need to rename
                                            base_name = entry.name.rsplit('.', 1)[0]
                                            extension = entry.name.rsplit('.', 1)[1] if '.' in entry.name else 'mp4'
                                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                            to_path = f"{DROPBOX_FOLDERS['rejected']}/{base_name}_perm_{timestamp}.{extension}"
                                        except:
                                            # File doesn't exist, proceed normally
                                            pass
                                        
                                        dropbox_manager.dbx.files_move_v2(entry.path_display, to_path)
                                        logger.info(f"Moved {entry.name} from {folder_name} to rejected (permanent)")
                                        
                                        videos_rejected.append({
                                            'filename': entry.name,
                                            'from_folder': folder_name
                                        })
                                    except Exception as e:
                                        logger.error(f"Error moving {entry.name} to rejected: {e}")
                                else:
                                    # Already in rejected folder
                                    videos_rejected.append({
                                        'filename': entry.name,
                                        'from_folder': 'rejected (already)'
                                    })
                    
                    # Check if there are more files
                    if not result.has_more:
                        break
                    result = dropbox_manager.dbx.files_list_folder_continue(result.cursor)
                    
            except Exception as e:
                logger.error(f"Error searching folder {folder_name}: {e}")
        
        # Cancel all active workflows for this task
        workflows_cancelled = []
        
        # Check in-memory workflows
        for workflow_id, workflow in list(approval_workflows.items()):
            if workflow.get('task_number') == task_number:
                workflows_cancelled.append((workflow_id, workflow))
                
                # Remove from cache and database
                if workflow_id in approval_workflows:
                    del approval_workflows[workflow_id]
                await delete_workflow_async(workflow_id)
        
        # Check database workflows
        try:
            all_pending_workflows = await get_all_pending_workflows_async()
            for workflow in all_pending_workflows:
                if workflow.get('task_number') == task_number:
                    workflow_id = workflow['workflow_id']
                    if workflow_id not in [w[0] for w in workflows_cancelled]:
                        workflows_cancelled.append((workflow_id, workflow))
                        await delete_workflow_async(workflow_id)
        except Exception as e:
            logger.error(f"Error checking database for workflows: {e}")
        
        # Update any active Slack messages
        for workflow_id, workflow in workflows_cancelled:
            try:
                stage = workflow.get('stage', '')
                
                if stage == 'reviewer' and workflow.get('reviewer_msg_ts'):
                    reviewer_channel = (await get_reviewer_channel()) or workflow.get('reviewer_id')
                    if reviewer_channel:
                        try:
                            await slack_client.chat_update(
                                channel=reviewer_channel,
                                ts=workflow['reviewer_msg_ts'],
                                text="⛔ This task has been permanently rejected",
                                blocks=[{
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": f"⛔ *Task Permanently Rejected*\n\nTask #{task_number} has been permanently rejected and archived."
                                    }
                                }]
                            )
                        except Exception as e:
                            logger.warning(f"Could not update reviewer message: {e}")
                
                elif stage == 'hos' and workflow.get('hos_msg_ts'):
                    hos_channel = workflow.get('hos_id')
                    if hos_channel:
                        try:
                            await slack_client.chat_update(
                                channel=hos_channel,
                                ts=workflow['hos_msg_ts'],
                                text="⛔ This task has been permanently rejected",
                                blocks=[{
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": f"⛔ *Task Permanently Rejected*\n\nTask #{task_number} has been permanently rejected and archived."
                                    }
                                }]
                            )
                        except Exception as e:
                            logger.warning(f"Could not update HoS message: {e}")
                            
            except Exception as e:
                logger.error(f"Error updating messages for workflow {workflow_id}: {e}")
        
        # Notify videographer about permanent rejection
        if videographer_id and videos_rejected:
            try:
                video_list = "\n".join([f"• `{v['filename']}` (from {v['from_folder']})" for v in videos_rejected])
                
                await slack_client.chat_postMessage(
                    channel=videographer_id,
                    text=f"⛔ Task #{task_number} has been permanently rejected",
                    blocks=[{
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"⛔ *Task Permanently Rejected*\n\nTask #{task_number} has been permanently rejected and archived.\n\n" +
                                   f"Brand: {task_data.get('Brand', 'N/A')}\n" +
                                   f"Reference: {task_data.get('Reference Number', 'N/A')}\n\n" +
                                   f"The following videos have been moved to the rejected folder:\n{video_list}\n\n" +
                                   "*This task will not count towards reviewer response time metrics.*"
                        }
                    }]
                )
            except Exception as e:
                logger.error(f"Error notifying videographer: {e}")
        
        # Archive any Trello card
        try:
            from trello_utils import get_trello_card_by_task_number, archive_trello_card
            card = await get_trello_card_by_task_number(task_number)
            if card:
                await asyncio.to_thread(archive_trello_card, card['id'])
                logger.info(f"Archived Trello card for permanently rejected Task #{task_number}")
            else:
                logger.warning(f"No Trello card found for permanently rejected Task #{task_number}")
        except Exception as e:
            logger.warning(f"Error archiving Trello card: {e}")
        
        logger.info(f"Permanent rejection completed for Task #{task_number}")
        
    except Exception as e:
        logger.error(f"Error in handle_permanent_rejection: {e}")
        raise


async def recover_pending_workflows():
    """Recover pending workflows from database on startup"""
    try:
        logger.info("Recovering pending approval workflows from database...")
        
        # Get all pending workflows from database
        pending_workflows = await get_all_pending_workflows_async()
        
        if not pending_workflows:
            logger.info("No pending workflows found to recover")
            return
        
        logger.info(f"Found {len(pending_workflows)} pending workflows to recover")
        
        # Restore to cache silently
        for workflow in pending_workflows:
            workflow_id = workflow.get('workflow_id')
            task_number = workflow.get('task_number')
            stage = workflow.get('stage', 'reviewer')
            
            # Restore to cache
            approval_workflows[workflow_id] = workflow
            
            logger.info(f"Recovered workflow {workflow_id} - Task #{task_number} - Stage: {stage}")
        
        logger.info(f"Successfully recovered {len(pending_workflows)} workflows")
        
    except Exception as e:
        logger.error(f"Error recovering pending workflows: {e}")
