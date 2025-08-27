# Admin Workflow for Managing People

## Overview
Only admins (Head of Department and Reviewer) can add, update, or remove people in the system. This includes videographers, salespeople, and updating Slack IDs.

## Workflow Process

### For New Users:
1. **User runs `/my_ids` command** - This shows their Slack information in a copyable format
2. **User copies their information** and sends it to an admin
3. **Admin updates the system** using the bot's tools

### For Admins:

#### Adding a new videographer:
```
"Add videographer John Doe with email john@example.com"
```

#### Adding a new salesperson:
```
"Add salesperson Jane Smith with email jane@example.com"
```

#### Updating Slack IDs:
When someone sends you their IDs from `/my_ids`, you can update them:
```
"Update Slack IDs for videographer John Doe: user ID U1234567890, channel ID C0987654321"
"Update Slack IDs for salesperson Jane Smith: user ID U9876543210"
"Update Slack IDs for reviewer: user ID U1111111111"
```

## Permissions
- **CREATE_PERMISSIONS**: Can create new design requests
- **EDIT_PERMISSIONS**: Can edit existing tasks
- **ADMIN_PERMISSIONS**: Can manage people and update Slack IDs (restricted to Head of Department and Reviewer)

## Example `/my_ids` Output
```
🆔 Your Slack Information

User Details:
• Name: John Doe
• Email: john.doe@example.com
• User ID: U1234567890

Channel Information:
• Channel: general
• Type: Public Channel
• Channel ID: C0987654321

📋 Copyable Format for Admin:
Name: John Doe
Email: john.doe@example.com
Slack User ID: U1234567890
Slack Channel ID: C0987654321

💡 Next Steps:
1. Copy the above information
2. Send it to your admin (Head of Department or Reviewer)
3. They will add you to the system with these IDs
```

## Why This Workflow?
- **Security**: Only authorized admins can modify the system
- **Accuracy**: Users provide their own IDs, reducing errors
- **Traceability**: All changes go through proper channels