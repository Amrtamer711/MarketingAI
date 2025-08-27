import os
import pytz
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).parent.parent / '.env')

# Set UAE timezone
UAE_TZ = pytz.timezone('Asia/Dubai')

# ========== CONFIGURATION ==========
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Excel file configuration
# Use /data/ for production (absolute path), data/ for local (relative path)
IS_PRODUCTION = os.environ.get("PORT") is not None  # Render sets PORT
DATA_DIR = "/data" if IS_PRODUCTION else "data"
EXCEL_FILE_PATH = f"{DATA_DIR}/design_requests.xlsx"

# Trello configuration
TRELLO_API_KEY = os.environ.get("TRELLO_API_KEY")
TRELLO_API_TOKEN = os.environ.get("TRELLO_API_TOKEN")
BOARD_NAME = "Amr - Tracker"

# Environment no longer needed - always use local Excel/DB

# Default history DB path
HISTORY_DB_PATH = f"{DATA_DIR}/history_logs.db"

# Email configuration
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "your-email@example.com")
APP_PSWD = os.getenv("APP_PSWD", "your-app-password")  # Keep for backward compatibility

# Email recipients
REVIEWER_EMAIL = os.getenv("REVIEWER_EMAIL", "reviewer@example.com")
HEAD_OF_DEPT_EMAIL = os.getenv("HEAD_OF_DEPT_EMAIL", "hod@example.com")
HEAD_OF_SALES_EMAIL = os.getenv("HEAD_OF_SALES_EMAIL", "hos@example.com")

# Dropbox credentials path
CREDENTIALS_PATH = Path(f"{DATA_DIR}/dropbox_creds.json")

# Videographer config path
VIDEOGRAPHER_CONFIG_PATH = f"{DATA_DIR}/videographer_config.json"

# Note: DROPBOX_FOLDERS is defined in video_upload_system.py as a dict mapping folder names to paths

# For testing: use 10 seconds instead of 5 PM deadline
TESTING_MODE = True
ESCALATION_DELAY_SECONDS = 10 if TESTING_MODE else 0  # In production, check time instead

WEEKEND_DAYS = {4, 5, 6}  # Friday, Saturday (UAE weekend)
CAMPAIGN_LOOKAHEAD_WORKING_DAYS = 10  # Working days before campaign to create tasks
PLANNING_OFFSET_DAYS = 1
VIDEO_TASK_OFFSET_WORKING_DAYS = 2

