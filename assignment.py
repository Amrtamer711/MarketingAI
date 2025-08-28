import pandas as pd
from datetime import datetime
from config import BOARD_NAME
from trello_utils import get_board_id_by_name, create_card_on_board, count_working_days, calculate_filming_date, will_be_available_by_date, get_leave_dates, LOCATION_TO_VIDEOGRAPHER, get_best_videographer_for_balancing   
from db_utils import select_all_tasks as db_select_all_tasks
from db_utils import update_task_by_number as db_update_task_by_number


# ========== MAIN CAMPAIGN PROCESSING ==========
def check_and_assign_tasks():
    """Check for unassigned campaigns and create Trello tasks"""
    try:
        # Read live tasks from DB
        rows = db_select_all_tasks()
        df = pd.DataFrame(rows)
        if 'task_number' in df.columns:
            df.rename(columns={'task_number': 'Task #'}, inplace=True)
        print(f"📊 Found {len(df)} total design requests (DB)")
        
        # Convert columns to string type to avoid dtype warnings
        df['Videographer'] = df['Videographer'].astype('object')
        if 'Video Filename' in df.columns:
            df['Video Filename'] = df['Video Filename'].astype('object')
        
        # Get current date
        today = datetime.now().date()
        
        # Filter for unassigned campaigns
        unassigned = df[df['Status'] == 'Not assigned yet'].copy()
        print(f"📋 Found {len(unassigned)} unassigned campaigns")
        
        # Track assignments made
        assignments_made = []
        
        # Track campaign letters for duplicate ref/location/brand/month combinations
        campaign_letters = {}
        
        for idx, row in unassigned.iterrows():
            try:
                # Parse campaign date (expecting dd-mm-yyyy format)
                campaign_date = pd.to_datetime(row['Campaign Start Date'], dayfirst=True).date()
                
                # Calculate working days until campaign
                working_days_until = count_working_days(datetime.combine(today, datetime.min.time()), 
                                                       datetime.combine(campaign_date, datetime.min.time()))
                
                # Check if within 10 working days
                if working_days_until <= 10 and working_days_until >= 0:
                    location = row['Location']
                    
                    # Get board ID for checking workloads
                    board_id = get_board_id_by_name(BOARD_NAME)
                    
                    # Parse campaign end date
                    campaign_end_date = None
                    if 'Campaign End Date' in row and pd.notna(row['Campaign End Date']):
                        try:
                            campaign_end_date = pd.to_datetime(row['Campaign End Date'], dayfirst=True).date()
                        except:
                            pass
                    
                    # First calculate filming date
                    filming_date_str = row.get('Filming Date', '')
                    if filming_date_str:
                        video_date = pd.to_datetime(filming_date_str, dayfirst=True).date()
                    else:
                        # Calculate filming date using new rules
                        video_date = calculate_filming_date(campaign_date, campaign_end_date).date()
                    
                    # Get primary assigned person for location
                    primary_person = LOCATION_TO_VIDEOGRAPHER.get(location)
                    assigned_person = None
                    
                    # Check if primary person will be available by filming date
                    if primary_person:
                        if not will_be_available_by_date(board_id, primary_person, video_date):
                            # Type 1: On Leave Load Balancing
                            start_date, end_date = get_leave_dates(board_id, primary_person)
                            if start_date and end_date:
                                print(f"  ⚠️ {primary_person} is on leave from {start_date.strftime('%d-%m-%Y')} to {end_date.strftime('%d-%m-%Y')} (filming: {video_date})")
                            elif end_date:
                                print(f"  ⚠️ {primary_person} is on leave until {end_date.strftime('%d-%m-%Y')} (filming: {video_date})")
                            elif start_date:
                                print(f"  ⚠️ {primary_person} is on leave starting {start_date.strftime('%d-%m-%Y')} (filming: {video_date})")
                            else:
                                print(f"  ⚠️ {primary_person} is on leave (no dates specified) for {location}")
                            print(f"  🔄 Finding alternative videographer (Leave Load Balancing)...")
                            assigned_person, workloads = get_best_videographer_for_balancing(board_id, primary_person, "leave", target_date=video_date)
                            if assigned_person:
                                print(f"  ✅ Reassigned to {assigned_person} (due to leave)")
                        else:
                            # No workload balancing - assign to primary person
                            assigned_person = primary_person
                    else:
                        # No primary person assigned, use load balancing
                        print(f"  ⚠️ No default videographer for location '{location}'")
                        print(f"  🔄 Using load balancing to assign...")
                        assigned_person, workloads = get_best_videographer_for_balancing(board_id, target_date=video_date)
                    
                    if assigned_person:
                        # Calculate editing deadline (3 working days after filming)
                        from trello_utils import add_working_days
                        editing_deadline = add_working_days(video_date, 3)
                        
                        # Create Trello card
                        task_number = row.get('Task #', '')
                        card_title = f"Task #{task_number}: 📸 {row['Brand']} - {row['Reference Number']}"
                        card_description = (
                            f"**Task #:** {task_number}\n\n"
                            f"**Campaign Details:**\n"
                            f"• Brand: {row['Brand']}\n"
                            f"• Campaign Start Date: {campaign_date.strftime('%d-%m-%Y')}\n"
                            f"• Campaign End Date: {campaign_end_date.strftime('%d-%m-%Y') if campaign_end_date else 'N/A'}\n"
                            f"• Location: {location}\n"
                            f"• Reference: {row['Reference Number']}\n"
                            f"• Sales Person: {row.get('Sales Person', 'Unknown')}\n"
                            f"• Submitted by: {row.get('Submitted By', 'Unknown')}\n\n"
                            f"**Task:** Conduct videography session on scheduled filming date\n"
                            f"**Filming Date (Start):** {video_date.strftime('%d-%m-%Y')}\n"
                            f"**Editing Deadline (Due):** {editing_deadline.strftime('%d-%m-%Y')}"
                        )
                        
                        # Add note if reassigned
                        if primary_person and assigned_person != primary_person:
                            card_description += f"\n\n**Note:** Reassigned from {primary_person} (on leave)"
                        
                        # Create the card with filming as start date and editing deadline as due date
                        card = create_card_on_board(
                            board_name=BOARD_NAME,
                            card_title=card_title,
                            card_description=card_description,
                            due_date=editing_deadline,  # Main due date is editing deadline
                            list_name=assigned_person,
                            start_date=video_date  # Start date is filming date
                        )
                        
                        # Add checklist with filming and editing dates
                        if card and card.get('id'):
                            from trello_utils import create_checklist_with_dates
                            create_checklist_with_dates(card['id'], video_date)
                        
                        # Update status and videographer in DB
                        db_update_task_by_number(int(task_number), {
                            'Status': f'Assigned to {assigned_person}',
                            'Videographer': assigned_person
                        })
                        
                        assignments_made.append({
                            'brand': row['Brand'],
                            'reference': row['Reference Number'],
                            'location': location,
                            'assigned_to': assigned_person,
                            'video_date': video_date,
                            'trello_url': card.get('shortUrl', 'N/A'),
                            'reassigned': bool(primary_person and assigned_person != primary_person)
                        })
                        
                        print(f"✅ Assigned {row['Brand']} ({row['Reference Number']}) to {assigned_person}")
                        print(f"   📅 Campaign: {campaign_date.strftime('%d-%m-%Y')} → Videography: {video_date.strftime('%d-%m-%Y')}")
                        print(f"   🔗 Trello: {card.get('shortUrl', 'N/A')}")
                    else:
                        print(f"❌ No available videographer for {row['Brand']} ({row['Reference Number']}) - all are on leave or unavailable")
                
            except Exception as e:
                print(f"❌ Error processing row {idx}: {e}")
                continue
        
        if not assignments_made:
            print("\n📝 No new assignments made")
        
        return assignments_made
        
    except Exception as e:
        print(f"❌ Error in task assignment: {e}")
        return []


# ========== EXECUTION ==========
if __name__ == "__main__":
    print("🚀 Starting task assignment check...")
    print(f"📍 Location mapping: {LOCATION_TO_VIDEOGRAPHER}")
    print()
    
    assignments = check_and_assign_tasks()
    
    print("\n📊 Summary:")
    print(f"Total assignments made: {len(assignments)}")
    if assignments:
        print("\nAssignments:")
        for a in assignments:
            print(f"  • {a['brand']} → {a['assigned_to']} (Video on {a['video_date'].strftime('%d-%m-%Y')})")
