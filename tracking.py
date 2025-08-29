import pandas as pd
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from config import EMAIL_SENDER, APP_PSWD, REVIEWER_EMAIL, HEAD_OF_DEPT_EMAIL, TESTING_MODE, ESCALATION_DELAY_SECONDS
from dropbox_utils import init_dropbox, search_file_in_dropbox, get_latest_version, get_status_from_folder
from management import load_videographer_config
from trello_utils import set_trello_due_complete, archive_trello_card, get_trello_card_by_task_number
from utils import check_time_conditions, check_raw_folder_deadline

# Load configuration and extract emails
videographer_config = load_videographer_config()
VIDEOGRAPHER_EMAILS = {name: info["email"] for name, info in videographer_config["videographers"].items()}

def send_email(to_emails, subject, body):
    """Send email notification using SMTP with app password"""
    try:
        # Create message
        message = MIMEMultipart()
        message['to'] = ', '.join(to_emails) if isinstance(to_emails, list) else to_emails
        message['from'] = EMAIL_SENDER
        message['subject'] = subject
        
        # Attach the body
        msg = MIMEText(body, 'plain')
        message.attach(msg)
        
        # Send using SMTP
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, APP_PSWD)
            server.send_message(message)
        
        print(f"✅ Email sent to: {message['to']}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False


def main(test_date=None):
    """Main function to track missing video submissions
    
    Workflow:
    1. Videos must be submitted to /Site Videos/Raw by 12pm/5pm (1 working day after filming)
    2. Videos must move from Raw to Pending/Submitted/Accepted within 3 working days
    3. Videos are considered overdue if not completed within 15 working days total
    4. Videos in Returned folder need revision (sales didn't approve from Accepted)
    """
    # Use test date if provided, otherwise use current time
    current_time = test_date if test_date else datetime.now()
    
    print("🎬 Video Submission Tracker")
    print(f"📅 Current time: {current_time.strftime('%d-%m-%Y %H:%M:%S')}")
    print(f"🧪 Testing mode: {'ON' if TESTING_MODE else 'OFF'}")
    if test_date:
        print(f"🧪 Using test date: {test_date.strftime('%d-%m-%Y')}")
    print()
    
    # Initialize Dropbox
    dbx = init_dropbox()
    if not dbx:
        return
    
    # Read tasks from database
    try:
        from db_utils import select_all_tasks, tasks_to_dataframe
        rows = select_all_tasks()
        df = tasks_to_dataframe(rows)
        print(f"📊 Found {len(df)} total tasks\n")
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return
    
    # Filter for assigned tasks
    assigned_tasks = df[df['Status'].str.startswith('Assigned to')].copy()
    print(f"📋 Found {len(assigned_tasks)} assigned tasks\n")
    
    # Track tasks that need notifications
    tasks_needing_notification = []
    tasks_for_escalation = []
    
    for idx, row in assigned_tasks.iterrows():
        task_number = row['Task #']
        filming_date = row['Filming Date']
        videographer = row.get('Videographer', '')
        
        print(f"🔍 Checking Task #{task_number} - {row['Brand']} ({row['Reference Number']})")
        print(f"   Filming date: {filming_date}")
        print(f"   Videographer: {videographer}")
        
        # Check time conditions
        past_noon, past_five_pm, working_days_overdue = check_time_conditions(filming_date, current_time)
        
        # In testing mode, skip time checks
        if not past_noon and not TESTING_MODE:
            print(f"   ⏰ Not yet 12 PM the day after filming\n")
            continue
        
        # Check if card exists in Trello
        trello_card = get_trello_card_by_task_number(task_number)
        
        if not trello_card:
            print(f"   ❌ Task no longer exists in Trello - flagging to reviewer\n")
            # Add to notification list for immediate review
            task_info = {
                'task_number': task_number,
                'brand': row['Brand'],
                'reference': row['Reference Number'],
                'filming_date': filming_date,
                'videographer': videographer,
                'location': row['Location'],
                'missing_from_trello': True
            }
            tasks_needing_notification.append(task_info)
            continue
        
        # Generate expected video filename
        # Using the video naming convention from task_assignment.py
        ref_number = row['Reference Number'].replace(' ', '')
        location_clean = row['Location'].replace(' ', '')
        brand_clean = row['Brand'].replace(' ', '')
        sales_person_raw = row.get('Sales Person', '')
        sales_person_clean = sales_person_raw.replace(' ', '') if sales_person_raw else ''
        
        try:
            campaign_date = pd.to_datetime(row['Campaign Start Date'])
            month_year = campaign_date.strftime('%b%y')
        except:
            month_year = "Unknown"
        
        # Search for video file patterns
        search_pattern = f"{ref_number}_{location_clean}_{brand_clean}_{month_year}_{videographer}_{sales_person_clean}"
        print(f"   🔍 Searching for files matching: {search_pattern}*")
        
        # Search in Dropbox
        found_files = search_file_in_dropbox(dbx, search_pattern)
        
        if found_files:
            print(f"   ✅ Found {len(found_files)} matching file(s):")
            for f in found_files:
                print(f"      - {f['file']} in {f['folder']}")
            
            # Find the latest version
            latest_file = get_latest_version(found_files)
            if latest_file:
                print(f"   📝 Latest version: {latest_file['file']} in {latest_file['folder']}")
                
                # Update status based on folder
                new_status = get_status_from_folder(latest_file['folder'])
                # Update status in database
                from db_utils import update_task_by_number
                update_task_by_number(task_number, {'Status': new_status})
                df.at[idx, 'Status'] = new_status
                print(f"   ✅ Updated status to: {new_status}")
                
                # Reflect status in Trello (tick/untick/archive)
                try:
                    trello_card = get_trello_card_by_task_number(task_number)
                    if trello_card:
                        folder = latest_file['folder']
                        if folder == "/Site Videos/Submitted to Sales":
                            set_trello_due_complete(trello_card['id'], True)
                        elif folder == "/Site Videos/Returned":
                            set_trello_due_complete(trello_card['id'], False)
                        elif folder == "/Site Videos/Accepted":
                            archive_trello_card(trello_card['id'])
                    else:
                        print("   ⚠️ Trello card not found for this task")
                except Exception as e:
                    print(f"   ⚠️ Error updating Trello card: {e}")
                
                # Special handling for videos in Raw folder
                if latest_file['folder'] == "/Site Videos/Raw":
                    is_raw_overdue, days_in_raw = check_raw_folder_deadline(filming_date, current_time)
                    if is_raw_overdue:
                        print(f"   🚨 VIDEO STUCK IN RAW: {days_in_raw} working days in raw folder (limit: 3)")
                        # Add to escalation for being stuck in raw
                        task_info = {
                            'task_number': task_number,
                            'brand': row['Brand'],
                            'reference': row['Reference Number'],
                            'filming_date': filming_date,
                            'videographer': videographer,
                            'location': row['Location'],
                            'stuck_in_raw': True,
                            'days_in_raw': days_in_raw,
                            'current_folder': latest_file['folder']
                        }
                        tasks_for_escalation.append(task_info)
                
                # Check if overdue (15 working days)
                elif new_status != "Done" and working_days_overdue > 15:
                    print(f"   ⚠️ OVERDUE: {working_days_overdue} working days since filming")
                    df.at[idx, 'Status'] = "Overdue"
                    # Add to escalation list
                    task_info = {
                        'task_number': task_number,
                        'brand': row['Brand'],
                        'reference': row['Reference Number'],
                        'filming_date': filming_date,
                        'videographer': videographer,
                        'location': row['Location'],
                        'overdue': True,
                        'days_overdue': working_days_overdue
                    }
                    tasks_for_escalation.append(task_info)
            print()
        else:
            print(f"   ⚠️ No files found matching pattern")
            
            # Check if overdue
            if working_days_overdue > 15:
                print(f"   ⚠️ OVERDUE: {working_days_overdue} working days since filming")
                df.at[idx, 'Status'] = "Overdue"
            
            # Add to notification list
            task_info = {
                'task_number': task_number,
                'brand': row['Brand'],
                'reference': row['Reference Number'],
                'filming_date': filming_date,
                'videographer': videographer,
                'location': row['Location']
            }
            
            if working_days_overdue > 15:
                task_info['overdue'] = True
                task_info['days_overdue'] = working_days_overdue
            
            # Decide whether to notify or escalate
            if past_five_pm or working_days_overdue > 15:
                tasks_for_escalation.append(task_info)
                print(f"   🚨 Added to escalation list\n")
            else:
                tasks_needing_notification.append(task_info)
                print(f"   📧 Added to notification list\n")
    
    # Send notifications for missing files (12 PM deadline)
    if tasks_needing_notification:
        print(f"\n📧 Preparing notifications for {len(tasks_needing_notification)} missing videos...")
        
        # Group tasks by videographer
        videographer_tasks = {}
        missing_from_trello = []
        
        for task in tasks_needing_notification:
            if task.get('missing_from_trello'):
                missing_from_trello.append(task)
            else:
                videographer = task['videographer']
                if videographer not in videographer_tasks:
                    videographer_tasks[videographer] = []
                videographer_tasks[videographer].append(task)
        
        # Send one email per videographer with all their missing videos
        for videographer, tasks in videographer_tasks.items():
            videographer_email = VIDEOGRAPHER_EMAILS.get(videographer, f"{videographer.lower()}@example.com")
            subject = f"Missing Video Submissions - {len(tasks)} videos pending"
            
            body = f"""Dear {videographer},

This is a reminder that the following videos have not been submitted:

"""
            for task in tasks:
                body += f"""Task #{task['task_number']}:
- Brand: {task['brand']}
- Reference: {task['reference']}
- Location: {task['location']}
- Filming Date: {task['filming_date']}

"""
            
            body += """Please submit these videos to Dropbox as soon as possible.

If any videos have already been submitted, please verify they're in the correct folder with the proper naming convention.

Best regards,
Video Tracking System"""
            
            send_email(videographer_email, subject, body)
            print(f"✅ Sent notification to {videographer} for {len(tasks)} missing videos")
        
        # Send summary email to reviewer
        if videographer_tasks or missing_from_trello:
            subject = f"Video Submission Status Report - {len(tasks_needing_notification)} videos missing"
            body = """Dear Reviewer,

Here's the summary of missing video submissions:

"""
            if videographer_tasks:
                body += "**Missing Videos by Videographer:**\n\n"
                for videographer, tasks in videographer_tasks.items():
                    body += f"{videographer} ({len(tasks)} videos):\n"
                    for task in tasks:
                        body += f"  - Task #{task['task_number']}: {task['brand']} ({task['reference']}), Filmed: {task['filming_date']}\n"
                    body += "\n"
            
            if missing_from_trello:
                body += "**URGENT - Tasks Missing from Trello:**\n\n"
                for task in missing_from_trello:
                    body += f"""Task #{task['task_number']}:
- Brand: {task['brand']}
- Reference: {task['reference']}
- Location: {task['location']}
- Videographer: {task['videographer']}
- Filming Date: {task['filming_date']}

This task was removed from Trello but is still marked as "Assigned" in our system.
Please investigate and ensure it follows the correct workflow.

"""
            
            body += """Best regards,
Video Tracking System"""
            
            send_email(REVIEWER_EMAIL, subject, body)
            print(f"✅ Sent summary notification to reviewer")
    
    # In testing mode, wait and then escalate all notification tasks
    if TESTING_MODE and tasks_needing_notification:
        print(f"\n⏰ Waiting {ESCALATION_DELAY_SECONDS} seconds before escalation (testing mode)...")
        time.sleep(ESCALATION_DELAY_SECONDS)
        # Move all notification tasks to escalation
        tasks_for_escalation.extend(tasks_needing_notification)
        print(f"\n🚨 TEST MODE: Escalating all {len(tasks_needing_notification)} tasks to Head of Department...")
    
    # Handle escalations (5 PM deadline or testing mode)
    if tasks_for_escalation:
        print(f"\n🚨 Escalating {len(tasks_for_escalation)} issues to Head of Department...")
        # Separate different types of issues
        overdue_tasks = [t for t in tasks_for_escalation if t.get('overdue')]
        stuck_in_raw_tasks = [t for t in tasks_for_escalation if t.get('stuck_in_raw')]
        regular_escalations = [t for t in tasks_for_escalation if not t.get('overdue') and not t.get('stuck_in_raw')]
        
        # Group by videographer for summary
        by_videographer = {}
        by_videographer_overdue = {}
        by_videographer_raw = {}
        
        for task in regular_escalations:
            videographer = task['videographer']
            if videographer not in by_videographer:
                by_videographer[videographer] = []
            by_videographer[videographer].append(task)
            
        for task in overdue_tasks:
            videographer = task['videographer']
            if videographer not in by_videographer_overdue:
                by_videographer_overdue[videographer] = []
            by_videographer_overdue[videographer].append(task)
        
        for task in stuck_in_raw_tasks:
            videographer = task['videographer']
            if videographer not in by_videographer_raw:
                by_videographer_raw[videographer] = []
            by_videographer_raw[videographer].append(task)
        
        # Send escalation email to Head of Department
        subject = f"URGENT: {len(tasks_for_escalation)} Missing/Overdue Video Submissions"
        body = f"""Dear Head of Department,

The following video submissions require immediate attention:

"""
        
        if stuck_in_raw_tasks:
            body += f"\n🔴 VIDEOS STUCK IN RAW FOLDER (3+ WORKING DAYS):\n"
            body += "These videos need to be processed immediately!\n"
            for videographer, tasks in by_videographer_raw.items():
                body += f"\n{videographer} ({len(tasks)} stuck):\n"
                for task in tasks:
                    body += f"  - Task #{task['task_number']}: {task['brand']} ({task['reference']}), Filmed: {task['filming_date']} - {task['days_in_raw']} DAYS IN RAW\n"
        
        if overdue_tasks:
            body += f"\n🚨 OVERDUE VIDEOS (15+ WORKING DAYS):\n"
            for videographer, tasks in by_videographer_overdue.items():
                body += f"\n{videographer} ({len(tasks)} overdue):\n"
                for task in tasks:
                    body += f"  - Task #{task['task_number']}: {task['brand']} ({task['reference']}), Filmed: {task['filming_date']} - {task['days_overdue']} DAYS OVERDUE\n"
        
        if regular_escalations:
            body += f"\n⚠️ MISSING VIDEOS (Past 5 PM Deadline):\n"
            for videographer, tasks in by_videographer.items():
                body += f"\n{videographer} ({len(tasks)} missing):\n"
                for task in tasks:
                    body += f"  - Task #{task['task_number']}: {task['brand']} ({task['reference']}), Filmed: {task['filming_date']}\n"
        
        body += f"""

Total issues: {len(tasks_for_escalation)} ({len(stuck_in_raw_tasks)} stuck in raw, {len(overdue_tasks)} overdue, {len(regular_escalations)} missing)

Please follow up with the videographers immediately.

Best regards,
Video Tracking System"""
        
        # Send to Head of Department
        send_email(HEAD_OF_DEPT_EMAIL, subject, body)
        print(f"✅ Sent escalation to Head of Department")
        
        # Send to Reviewer with same information
        send_email(REVIEWER_EMAIL, subject, body)
        print(f"✅ Sent escalation copy to Reviewer")
        
        # Send individual emails to each videographer about their specific tasks
        all_videographers = set(list(by_videographer.keys()) + list(by_videographer_overdue.keys()) + list(by_videographer_raw.keys()))
        for videographer in all_videographers:
            videographer_email = VIDEOGRAPHER_EMAILS.get(videographer, f"{videographer.lower()}@example.com")
            
            # Collect this videographer's tasks
            v_tasks = []
            v_overdue = []
            v_raw = []
            
            if videographer in by_videographer:
                v_tasks = by_videographer[videographer]
            if videographer in by_videographer_overdue:
                v_overdue = by_videographer_overdue[videographer]
            if videographer in by_videographer_raw:
                v_raw = by_videographer_raw[videographer]
            
            total_tasks = len(v_tasks) + len(v_overdue) + len(v_raw)
            subject = f"URGENT: {total_tasks} Video Submissions Required - Management Escalation"
            
            body = f"""Dear {videographer},

This is an urgent escalation notice. The following videos have not been submitted and management has been notified:

"""
            if v_raw:
                body += "🔴 VIDEOS STUCK IN RAW FOLDER (3+ working days):\n\n"
                for task in v_raw:
                    body += f"""Task #{task['task_number']}:
- Brand: {task['brand']}
- Reference: {task['reference']}
- Location: {task['location']}
- Filming Date: {task['filming_date']}
- Days in Raw: {task['days_in_raw']} working days

THIS VIDEO MUST BE PROCESSED IMMEDIATELY!

"""
            
            if v_overdue:
                body += "🚨 OVERDUE VIDEOS (15+ working days):\n\n"
                for task in v_overdue:
                    body += f"""Task #{task['task_number']}:
- Brand: {task['brand']}
- Reference: {task['reference']}
- Location: {task['location']}
- Filming Date: {task['filming_date']}
- Days Overdue: {task['days_overdue']} working days

"""
            
            if v_tasks:
                body += "⚠️ MISSING VIDEOS (Past 5 PM deadline):\n\n"
                for task in v_tasks:
                    body += f"""Task #{task['task_number']}:
- Brand: {task['brand']}
- Reference: {task['reference']}
- Location: {task['location']}
- Filming Date: {task['filming_date']}

"""
            
            body += """These videos must be submitted immediately. Management has been notified of this delay.

If any videos have already been submitted, please notify your supervisor immediately with proof of submission.

Best regards,
Video Tracking System"""
            
            send_email(videographer_email, subject, body)
            print(f"✅ Sent escalation to {videographer} for {total_tasks} videos")
    
    # Save updated Excel file with status changes
    # Database is already updated throughout the process
    print("\n✅ Updated Excel file with status changes")
    
    print(f"\n📊 Summary:")
    print(f"   - Tasks checked: {len(assigned_tasks)}")
    print(f"   - Missing videos (notified): {len(tasks_needing_notification)}")
    print(f"   - Missing videos (escalated): {len(tasks_for_escalation)}")

if __name__ == "__main__":
    import sys
    
    # Check for test date argument
    test_date = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-date" and len(sys.argv) > 2:
            try:
                # Parse date
                date_str = sys.argv[2]
                test_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                # Check for optional time parameter
                if len(sys.argv) > 3:
                    time_str = sys.argv[3]
                    if time_str == "12pm":
                        test_date = test_date.replace(hour=12, minute=0, second=0)
                    elif time_str == "5pm":
                        test_date = test_date.replace(hour=17, minute=0, second=0)
                    else:
                        # Try to parse as HH:MM
                        try:
                            hour, minute = map(int, time_str.split(':'))
                            test_date = test_date.replace(hour=hour, minute=minute, second=0)
                        except:
                            print(f"❌ Invalid time format. Use '12pm', '5pm', or 'HH:MM'")
                            sys.exit(1)
                
                print(f"🧪 TEST MODE: Using {test_date.strftime('%d-%m-%Y %H:%M:%S')}")
                print()
            except ValueError:
                print("❌ Invalid date format. Use YYYY-MM-DD")
                sys.exit(1)
    
    main(test_date)