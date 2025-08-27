# VideoCritique Bot

A Slack bot for managing video design requests, task assignments, and approval workflows.

## Features

- 📝 Design request creation and tracking
- 🎥 Video upload and approval workflow
- 📋 Trello integration for task management
- 📧 Email notifications
- 🤖 AI-powered request parsing
- 📊 Excel-based data storage with concurrency controls
- 🔄 Two-stage approval process (Reviewer → Sales)
- 💾 SQLite-based historical data archival

## Deployment on Render

### Prerequisites

1. **Slack App Setup**:
   - Create a Slack app at https://api.slack.com/apps
   - Configure OAuth scopes
   - Install to your workspace
   - Get Bot User OAuth Token and Signing Secret

2. **Trello Setup**:
   - Create a Trello board
   - Get API Key and Token from https://trello.com/app-key

3. **Dropbox Setup**:
   - Create a Dropbox app
   - Get refresh token and credentials
   - Save as `dropbox_creds.json`

4. **OpenAI Setup**:
   - Get API key from https://platform.openai.com


### Environment Variables

See [ENVIRONMENT.md](ENVIRONMENT.md) for detailed environment variables and configuration documentation.

### Deployment Steps

1. **Fork/Clone Repository**:
   ```bash
   git clone https://github.com/yourusername/videocritique-bot.git
   cd videocritique-bot
   ```

2. **Prepare Credentials**:
   - Add `dropbox_creds.json` to your repo (encrypted) or upload via Render shell
   - Ensure `videographer_config.json` is configured

3. **Deploy to Render**:
   - Connect your GitHub repo to Render
   - Render will use `render.yaml` for configuration
   - Service will auto-deploy on push

4. **Post-Deployment**:
   - Upload `dropbox_creds.json` to `/data/` directory (production uses absolute path)
   - Verify Excel file is created at `/data/design_requests.xlsx`
   - Test Slack integration

### Manual Setup (if needed)

1. **SSH into Render instance**:
   ```bash
   # Upload credentials
   scp dropbox_creds.json your-service:/data/
   ```

2. **Initialize data**:
   ```bash
   python startup.py
   ```

## Data Storage

### Excel-Based Storage

The bot uses Excel files for data storage with built-in concurrency controls:

- **Task Storage**: All design requests stored in:
  - Local development: `data/design_requests.xlsx`
  - Production: `/data/design_requests.xlsx`
- **History Archive**: Completed tasks archived in SQLite database
- **File Locking**: Prevents concurrent write conflicts using fcntl locks
- **Queue System**: Write operations are queued, never lost

### Slack Configuration

1. **Event Subscriptions**:
   - Request URL: `https://your-service.onrender.com/slack/events`
   - Subscribe to events:
     - `message.channels`
     - `message.im`
     - `app_mention`

2. **Interactivity & Shortcuts**:
   - Request URL: `https://your-service.onrender.com/slack/events`

3. **Slash Commands** (optional):
   - `/help_design` → `https://your-service.onrender.com/slack/slash-commands`
   - `/log_campaign` → Same URL
   - `/upload_video` → Same URL

### Cron Jobs

The `assignment.py` script runs daily at 9 AM UTC (1 PM UAE) to:
- Check for campaigns starting within 10 working days
- Assign tasks to videographers
- Create Trello cards
- Send notifications

### Monitoring

- Health check: `https://your-service.onrender.com/api/health`
- Logs: Available in Render dashboard
- Metrics: Monitor via Render dashboard

### Troubleshooting

1. **Bot not responding**:
   - Check Slack event subscriptions URL
   - Verify BOT_USER_ID is correct
   - Check logs for errors

2. **File not found errors**:
   - Ensure data directory exists:
     - Local: `data/` (relative path)
     - Production: `/data/` (absolute path)
   - Run `python startup.py` to create directories
   - Check file permissions

3. **Token refresh errors**:
   - Verify `dropbox_creds.json` exists in data directory
   - Check file has `last_refresh` field
   - Run `python migrate_dropbox_credentials.py`

4. **"Reviewer channel not configured" error**:
   - Ensure `videographer_config.json` has a `reviewer` section with `slack_channel_id`
   - See ENVIRONMENT.md for configuration file format
   - Run `python startup.py` to create default config if missing

## Production Deployment Checklist

Before deploying to production, ensure:

- [ ] All environment variables set in Render
- [ ] Dropbox credentials file prepared in `data/` directory
- [ ] Videographer config updated with correct emails/names
- [ ] Slack app configured with correct URLs
- [ ] Test data cleared from Excel file
- [ ] Data directories properly configured

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env with your credentials

# Run locally
python app.py
```

Use ngrok for Slack webhooks:
```bash
ngrok http 3000
# Update Slack app with ngrok URL
```