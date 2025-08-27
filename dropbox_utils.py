import json
import requests
from config import CREDENTIALS_PATH, DROPBOX_FOLDERS
import dropbox

# Dropbox token refresh functions
def load_credentials():
    with open(CREDENTIALS_PATH, "r") as f:
        return json.load(f)

def save_credentials(data):
    with open(CREDENTIALS_PATH, "w") as f:
        json.dump(data, f)

def refresh_access_token():
    creds = load_credentials()
    response = requests.post("https://api.dropbox.com/oauth2/token", data={
        "grant_type": "refresh_token",
        "refresh_token": creds["refresh_token"],
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"]
    })
    
    if response.status_code == 200:
        new_token = response.json()["access_token"]
        creds["access_token"] = new_token
        save_credentials(creds)
        print("✅ Dropbox access token refreshed.")
        return new_token
    else:
        raise Exception(f"❌ Failed to refresh token: {response.text}")

def init_dropbox():
    """Initialize Dropbox client"""
    global DROPBOX_API_KEY
    try:
        # Refresh token if needed
        DROPBOX_API_KEY = refresh_access_token()
        dbx = dropbox.Dropbox(DROPBOX_API_KEY)
        # Test the connection
        dbx.users_get_current_account()
        return dbx
    except Exception as e:
        print(f"❌ Failed to initialize Dropbox: {e}")
        return None

def search_file_in_dropbox(dbx, filename):
    """Search for a file in specified Dropbox folders"""
    found_locations = []
    
    for folder in DROPBOX_FOLDERS:
        try:
            # List files in folder
            result = dbx.files_list_folder(folder)
            
            # Check all entries
            while True:
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        # Check if filename matches (case-insensitive)
                        if filename.lower() in entry.name.lower():
                            found_locations.append({
                                'folder': folder,
                                'file': entry.name,
                                'path': entry.path_display
                            })
                
                # Check if there are more files
                if not result.has_more:
                    break
                    
                result = dbx.files_list_folder_continue(result.cursor)
                
        except Exception as e:
            print(f"⚠️ Error searching in {folder}: {e}")
    
    return found_locations

def get_latest_version(found_files):
    """Find the latest version from a list of found files"""
    if not found_files:
        return None
    
    latest_file = None
    latest_version = 0
    
    for file_info in found_files:
        filename = file_info['file']
        # Extract version number (last _num before extension)
        try:
            # Split by underscore and get the last part before extension
            parts = filename.rsplit('.', 1)[0].split('_')
            if parts:
                version_str = parts[-1]
                if version_str.isdigit():
                    version = int(version_str)
                    if version > latest_version:
                        latest_version = version
                        latest_file = file_info
        except:
            continue
    
    return latest_file if latest_file else (found_files[0] if found_files else None)

def get_status_from_folder(folder):
    """Determine status based on which folder the file is in"""
    folder_to_status = {
        "/Site Videos/Raw": "Raw - Awaiting Processing",
        "/Site Videos/Pending": "Critique",
        "/Site Videos/Rejected": "Editing", 
        "/Site Videos/Submitted to Sales": "Submitted to Sales",
        "/Site Videos/Accepted": "Done",
        "/Site Videos/Returned": "Returned by Sales"  # Sales didn't approve, needs revision
    }
    return folder_to_status.get(folder, "Unknown")