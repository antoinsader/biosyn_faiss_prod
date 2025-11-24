import json
import os
import sys

def load_logs_metadata():
    log_json_path = os.path.join("logs", "logger_all.json")
    if not os.path.exists(log_json_path):
        print(f"Error: {log_json_path} not found.")
        return []
    
    try:
        with open(log_json_path, 'r') as f:
            logs = json.load(f)
        return logs
    except json.JSONDecodeError:
        print(f"Error: Failed to decode {log_json_path}.")
        return []

def list_logs(logs):
    print("\nAvailable Logs:")
    for i, log in enumerate(logs):
        name = log.get("training_log_name", "Unknown")
        is_finished = log.get("finished", False)
        status = "(finished)" if is_finished else "(not finished)"
        print(f"{i + 1} - {name} {status}")

def get_epoch_summaries(log_file_path):
    if not os.path.exists(log_file_path):
        print(f"Error: Log file {log_file_path} not found.")
        return

    print(f"\nExtracting Epoch Summaries from: {log_file_path}\n")
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    looking_for_message = False
    found_any = False
    
    # Header for the table
    print(f"{'Epoch Summary Message':<80}")
    print("-" * 80)

    for line in lines:
        if "▶ EVENT LOG :: [TRAIN] :: [Epoch summary] ::" in line:
            looking_for_message = True
            continue
        
        if looking_for_message and "Message     :" in line:
            # Extract message content
            parts = line.split("Message     :", 1)
            if len(parts) > 1:
                message = parts[1].strip()
                print(message)
                found_any = True
            looking_for_message = False
    
    if not found_any:
        print("No epoch summaries found.")

def main():
    logs = load_logs_metadata()
    if not logs:
        return

    while True:
        list_logs(logs)
        
        choice = input("\nSelect a log number (or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            break
        
        if not choice.isdigit():
            print("Invalid input. Please enter a number.")
            continue
        
        idx = int(choice) - 1
        if idx < 0 or idx >= len(logs):
            print("Invalid selection. Please choose a number from the list.")
            continue
        
        selected_log = logs[idx]
        log_file_path = selected_log.get("log_details_file")
        
        # Handle relative paths if necessary (assuming running from root)
        if log_file_path.startswith("./"):
             # It is already relative to root if running from root
             pass
        
        print(f"\nSelected: {selected_log.get('training_log_name')}")
        
        while True:
            print("\nOptions:")
            print("1. get table epochs")
            print("b. Back to log list")
            
            opt = input("Select an option: ").strip()
            
            if opt == '1':
                print("loading get table epochs")
                get_epoch_summaries(log_file_path)
            elif opt.lower() == 'b':
                break
            else:
                print("Invalid option.")

if __name__ == "__main__":
    main()
