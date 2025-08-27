# Environment Variables and Configuration

## Required Environment Variables

### Slack Configuration
```bash
# Bot OAuth Token from Slack App
SLACK_BOT_TOKEN=xoxb-your-bot-token

# Signing Secret from Slack App (for request verification)
SLACK_SIGNING_SECRET=your-signing-secret

# Bot User ID (optional but recommended for better filtering)
BOT_USER_ID=U1234567890
```

### OpenAI Configuration
```bash
# OpenAI API Key for AI parsing
OPENAI_API_KEY=sk-your-api-key
```

### Trello Configuration
```bash
# Trello API credentials
TRELLO_API_KEY=your-api-key
TRELLO_API_TOKEN=your-token

# Board name should match exactly in Trello
BOARD_NAME="Amr - Tracker"
```

### Email Configuration (for notifications)
```bash
# Gmail account with app-specific password
EMAIL_SENDER=your-email@gmail.com
APP_PSWD=your-app-specific-password

# Notification recipients
REVIEWER_EMAIL=reviewer@company.com
HEAD_OF_DEPT_EMAIL=hod@company.com  
HEAD_OF_SALES_EMAIL=hos@company.com
```

### Optional Environment Variables
```bash
# Redis URL for distributed event deduplication
REDIS_URL=redis://your-redis-url

# PORT is automatically set by Render/hosting platforms
# Used to detect production environment
PORT=3000

# Slack channels (optional - can be configured in videographer_config.json instead)
REVIEWER_SLACK_CHANNEL=reviewer-channel-id
SALES_SLACK_CHANNEL=sales-channel-id
```

## Configuration Files

### 1. `data/videographer_config.json`

This file contains mappings for videographers, sales people, locations, and reviewer/admin configurations.

```json
{
    "videographers": {
        "John Doe": {
            "email": "john@company.com",
            "slack_channel_id": "C1234567890",
            "slack_user_id": "U1234567890",
            "list_id": "trello-list-id"
        }
    },
    "sales_people": {
        "Jane Smith": "jane@company.com",
        "Bob Johnson": "bob@company.com"
    },
    "location_mappings": {
        "Abu Dhabi": ["Abu Dhabi", "AD", "AUH"],
        "Dubai": ["Dubai", "DXB", "DUB"],
        "Sharjah": ["Sharjah", "SHJ"]
    },
    "reviewer": {
        "name": "Reviewer Name",
        "email": "reviewer@company.com",
        "slack_channel_id": "C9876543210",
        "slack_user_id": "U9876543210"
    },
    "hod": {
        "name": "HOD Name",
        "email": "hod@company.com",
        "slack_user_id": "U5555555555"
    },
    "head_of_sales": {
        "name": "Head of Sales",
        "email": "hos@company.com",
        "slack_user_id": "U6666666666"
    }
}
```

### 2. `data/dropbox_creds.json`

Contains Dropbox OAuth credentials:

```json
{
    "access_token": "sl.xxx...",
    "refresh_token": "xxx...",
    "client_id": "your-client-id",
    "client_secret": "your-client-secret",
    "last_refresh": 1234567890.0
}
```

## Directory Structure

The bot uses different directories based on environment:

- **Local Development**: `data/` (relative path)
- **Production**: `/data/` (absolute path)

The following files are stored in the data directory:
- `design_requests.xlsx` - Main Excel file with all tasks
- `history_logs.db` - SQLite database for archived tasks
- `videographer_config.json` - Configuration mappings
- `dropbox_creds.json` - Dropbox OAuth credentials

## Production Setup

1. Set all required environment variables
2. Ensure `/data/` directory exists with write permissions
3. Upload configuration files to `/data/`:
   - `videographer_config.json`
   - `dropbox_creds.json`
4. Run `python startup.py` to initialize directories and files

## Local Development Setup

1. Create `.env` file with environment variables
2. Run `python startup.py` to create `data/` directory
3. Place configuration files in `data/` directory
4. Start the application with `python app.py`