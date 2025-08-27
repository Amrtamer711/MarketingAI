# User conversation history and pending states
user_history = {}  # user_id -> list of messages
pending_confirmations = {}  # user_id -> parsed data awaiting confirmation
pending_duplicate_confirmations = {}  # user_id -> data with duplicate reference
slash_command_responses = {}  # user_id -> response_url for slash commands (not used for data input)
pending_edits = {}  # user_id -> {"task_number": X, "current_data": {...}, "updates": {...}}
pending_deletes = {}  # user_id -> {"task_number": X, "task_data": {...}}
