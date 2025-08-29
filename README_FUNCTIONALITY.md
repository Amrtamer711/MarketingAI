# VideoCritique Bot - Complete Functionality Specification

## Overview
VideoCritique Bot is a Slack-integrated task management system for handling video production requests, assignments, and approval workflows for marketing campaigns.

## Core Functionality

### 1. Task Creation & Management

#### 1.1 Design Request Creation
- **Text Input**: Users can type campaign details directly in Slack
- **Image Upload**: Upload campaign briefs as images (PNG, JPG, etc.) - AI extracts details
- **Email Parsing**: Forward emails with campaign details - AI extracts information
- **Required Fields**:
  - Brand/Client name
  - Campaign start date
  - Campaign end date  
  - Reference number (unique identifier)
  - Location (must match predefined locations)
  - Sales person (must match predefined sales people)
- **Automatic Calculation**: Filming date calculated based on campaign dates
- **Duplicate Detection**: Warns if reference number already exists

#### 1.2 Task Editing
- Edit any field of an existing task by task number
- Real-time validation of location and sales person names
- Maintains edit history and timestamps
- Requires edit_task permission

#### 1.3 Task Deletion  
- Delete tasks by task number
- Archives deleted tasks to history database
- Removes associated Trello cards if assigned
- Requires delete_task permission

#### 1.4 Task Viewing
- View all tasks via `/export_data` command
- Export current tasks as Excel file
- Export completed tasks history as SQLite database
- Filter and search capabilities

### 2. Videographer Assignment System

#### 2.1 Automatic Assignment (Daily Cron Job - 8 AM UAE)
- Scans for unassigned tasks
- Assigns tasks when campaign is within 10 working days
- Location-based assignment (each location has primary videographer)
- Leave management - reassigns if videographer is on leave
- Load balancing - distributes tasks evenly when primary is unavailable
- Creates Trello cards with checklists and due dates

#### 2.2 Manual Assignment
- Reassign tasks between videographers
- Update Trello cards automatically
- Maintain assignment history

### 3. Video Upload & Approval Workflow

#### 3.1 Video Upload
- Videographers upload videos via `/upload_video` command
- Supports Dropbox integration
- Version control system
- Automatic status updates
- Movement tracking between folders

#### 3.2 Multi-Stage Approval
1. **Videographer Upload** → Status: Raw
2. **Move to Pending** → Status: Critique (awaiting reviewer)
3. **Reviewer Decision**:
   - Approve → Status: Submitted to Sales
   - Reject → Status: Editing (with rejection reason)
4. **Head of Sales Decision**:
   - Accept → Status: Done (archived)
   - Return → Status: Returned (back to videographer)

#### 3.3 Rejection Tracking
- Categorized rejection reasons:
  - Previous Artwork is Visible
  - Competitor Billboard Visible
  - Artwork Color is Incorrect
  - Artwork Order is Incorrect
  - Environment Too Dark
  - Ghost Effect
  - Lighting of the cladding
  - Other (with custom comments)
- AI-powered rejection classification
- Rejection history per version

### 4. Permission System

#### 4.1 Role-Based Access Control
- **Super Admin**: All permissions
- **Admin**: User management, task management
- **Head of Department (HOD)**: Task management, user management
- **Head of Sales**: Task creation/editing, final video approval
- **Reviewer**: Task viewing, video review approval
- **Videographers**: Video upload, task viewing
- **Sales People**: Task creation, task viewing

#### 4.2 Group Permissions
- Users can belong to multiple groups
- Inherit highest permission level
- Configured via `videographer_config.json`

### 5. Slack Integration

#### 5.1 Commands
- `/help` - Show available commands
- `/upload_video` - Start video upload workflow
- `/export_data` - Export current data
- Direct messages or mentions for task creation

#### 5.2 Interactive Elements
- Buttons for approve/reject decisions
- Confirmation dialogs
- Real-time status updates
- Threaded conversations for context

#### 5.3 Notifications
- Assignment notifications to videographers
- Approval request notifications
- Status change notifications
- Due date reminders

### 6. Data Management

#### 6.1 Database Structure
- **Live Tasks** (SQLite): Active tasks
- **Completed Tasks** (SQLite): Archived/completed tasks
- **Version History**: JSON tracking all changes
- **Timestamps**: Movement tracking between statuses

#### 6.2 Excel Export
- Generate Excel files on demand
- Formatted with headers
- All task details included
- Backward compatibility maintained

### 7. Dashboard & Reporting

#### 7.1 Web Dashboard
- Real-time task statistics
- Performance metrics by videographer
- Campaign timeline visualization
- Rejection analysis
- Task aging reports

#### 7.2 Analytics
- Average completion times
- Rejection rates by category
- Videographer workload distribution
- Campaign success metrics

### 8. Integration Points

#### 8.1 Trello Integration
- Automatic card creation on assignment
- Checklist with filming milestones
- Due date management
- Card movement between lists
- Description updates with task details

#### 8.2 Dropbox Integration
- Video file storage
- Folder organization by status
- Version control
- Automatic file movement

#### 8.3 Email Integration
- Email notifications for approvals
- Campaign brief parsing
- Status update emails

### 9. Configuration Management

#### 9.1 Videographer Configuration
- Manage videographer list
- Slack ID mappings
- Email addresses
- Active/inactive status

#### 9.2 Location Management  
- Location to videographer mappings
- Add/remove locations
- Reassignment handling

#### 9.3 Sales People Management
- Add/remove sales people
- Contact information
- Slack integration

### 10. Advanced Features

#### 10.1 Leave Management
- Track videographer availability
- Automatic task redistribution
- Leave calendar integration

#### 10.2 Deduplication
- Redis-based request deduplication
- Prevents spam and repeated processing
- 5-minute cooldown for identical requests

#### 10.3 Error Recovery
- Automatic retry mechanisms
- Graceful degradation
- Comprehensive error logging

#### 10.4 Performance Optimization
- Concurrent operation handling
- Database connection pooling
- Async processing throughout
- WAL mode for SQLite

## Technical Requirements

### API Endpoints
- `/slack/events` - Handle all Slack events
- `/slack/slash-commands` - Process slash commands
- `/slack/interactive` - Handle button clicks and interactions
- `/api/health` - Health check endpoint
- `/api/dashboard` - Dashboard data API
- `/internal/run-assignment` - Cron job endpoint
- `/` - Web dashboard

### Environment Variables
- Slack tokens (bot, app, signing secret)
- Database paths
- Dropbox API keys
- Trello API keys
- Email credentials
- Redis connection (optional)

### Security
- Slack request verification
- Permission-based access control
- Input validation and sanitization
- SQL injection prevention
- Rate limiting

### Deployment
- Render.com compatible
- Persistent disk support for databases
- Cron job configuration
- Environment-based configuration
- Automatic SSL/HTTPS

## Expected User Flows

### Creating a Task
1. User uploads image or types details
2. Bot extracts/confirms information
3. User confirms or edits details
4. Task created with unique number
5. Notification sent

### Video Production Flow
1. Task auto-assigned when due
2. Videographer notified via Slack
3. Videographer films and uploads
4. Reviewer approves/rejects
5. If approved, Head of Sales reviews
6. Final approval archives task

### Editing a Task
1. User types "edit task X"
2. Bot shows current details
3. User specifies changes
4. Bot validates and updates
5. Confirmation sent

This system provides a complete workflow automation for video production requests, from initial request through final delivery, with comprehensive tracking, permissions, and integration capabilities.