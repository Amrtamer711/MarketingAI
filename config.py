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
# Production detection - single source of truth
# We're in production if ANY of these are true:
# 1. Running on Render (has RENDER env var)
# 2. Has PORT env var (web services)
# 3. Explicitly set PRODUCTION=true
# 4. /data directory exists (Render disk mount)
IS_PRODUCTION = any([
    os.environ.get("RENDER") == "true",
    os.environ.get("PORT") is not None,
    os.environ.get("PRODUCTION") == "true",
    os.path.exists("/data")
])

# Data directory configuration
DATA_DIR = "/data" if IS_PRODUCTION else "data"

# Log the environment for debugging
if IS_PRODUCTION:
    print(f"🚀 Running in PRODUCTION mode - using {DATA_DIR}/")
else:
    print(f"💻 Running in LOCAL mode - using {DATA_DIR}/")

# Trello configuration
TRELLO_API_KEY = os.environ.get("TRELLO_API_KEY")
TRELLO_API_TOKEN = os.environ.get("TRELLO_API_TOKEN")
BOARD_NAME = "Amr - Tracker"

# Environment no longer needed - always use local Excel/DB

# All file paths use DATA_DIR
HISTORY_DB_PATH = f"{DATA_DIR}/history_logs.db"
VIDEOGRAPHER_CONFIG_PATH = f"{DATA_DIR}/videographer_config.json"
CREDENTIALS_PATH = Path(f"{DATA_DIR}/dropbox_creds.json")

# Email configuration
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "your-email@example.com")
APP_PSWD = os.getenv("APP_PSWD", "your-app-password")  # Keep for backward compatibility

# Email recipients
REVIEWER_EMAIL = os.getenv("REVIEWER_EMAIL", "reviewer@example.com")
HEAD_OF_DEPT_EMAIL = os.getenv("HEAD_OF_DEPT_EMAIL", "hod@example.com")
HEAD_OF_SALES_EMAIL = os.getenv("HEAD_OF_SALES_EMAIL", "hos@example.com")

# Note: DROPBOX_FOLDERS is defined in video_upload_system.py as a dict mapping folder names to paths

# For testing: use 10 seconds instead of 5 PM deadline
TESTING_MODE = True
ESCALATION_DELAY_SECONDS = 10 if TESTING_MODE else 0  # In production, check time instead

WEEKEND_DAYS = {4, 5, 6}  # Friday, Saturday (UAE weekend)
CAMPAIGN_LOOKAHEAD_WORKING_DAYS = 10  # Working days before campaign to create tasks
PLANNING_OFFSET_DAYS = 1
VIDEO_TASK_OFFSET_WORKING_DAYS = 2

