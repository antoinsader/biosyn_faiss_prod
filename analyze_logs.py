import json
import os
import sys
import subprocess

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
        result_encoder_dir = log.get("result_encoder_dir", "")
        status = "(finished)" if is_finished else "(not finished)"
        id = log.get("id", "")
        print(f"{i+1}. {name} {status} {result_encoder_dir}, id: {id}")

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

    epoch_data = {}
    current_event = None
    current_epoch = None

    for line in lines:
        line = line.strip()
        if "▶ EVENT LOG :: [TRAIN]" in line:
            # Parse event type and epoch
            parts = line.split("::")
            if len(parts) >= 4:
                event_type = parts[2].strip(" []")
                epoch_str = parts[3].strip()
                if "epoch=" in epoch_str:
                    try:
                        current_epoch = int(epoch_str.split("=")[1].strip())
                        current_event = event_type
                        if current_epoch not in epoch_data:
                            epoch_data[current_epoch] = {"summary": "", "recall": ""}
                    except ValueError:
                        current_event = None
                        current_epoch = None
            else:
                current_event = None
                current_epoch = None
            continue
        
        if current_event and current_epoch is not None and line.startswith("Message     :"):
            message = line.split(":", 1)[1].strip()
            if "Epoch summary" in current_event:
                epoch_data[current_epoch]["summary"] = message
            elif "Faiss recall" in current_event:
                # Handle both old "Faiss recall@k" and new "Faiss recall"
                epoch_data[current_epoch]["recall"] = message
            
            # Reset after finding message to avoid capturing multiple lines if not intended
            # (Assuming one message line per event for now based on log format)
            current_event = None 

    if not epoch_data:
        print("No epoch summaries found.")
        return

    # Header for the table
    print(f"{'Epoch':<6} | {'Faiss Recall':<40} | {'Epoch Summary Message':<80}")
    print("-" * 130)

    for epoch in sorted(epoch_data.keys()):
        data = epoch_data[epoch]
        summary = data["summary"]
        recall = data["recall"]
        if summary or recall:
            print(f"{epoch:<6} | {recall:<40} | {summary}")

def run_eval(result_encoder_dir):
    if not result_encoder_dir:
        print("Error: No result_encoder_dir found in log details.")
        return

    print(f"\nRunning Eval for: {result_encoder_dir}")
    cmd = [sys.executable, "eval.py", "--result_encoder_dir", result_encoder_dir]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running eval: {e}")
    except KeyboardInterrupt:
        print("\nEval interrupted.")

def run_inference(result_encoder_dir):
    if not result_encoder_dir:
        print("Error: No result_encoder_dir found in log details.")
        return

    mention = input("Enter mention to normalize: ").strip()
    if not mention:
        print("Error: Mention cannot be empty.")
        return
        
    topk_str = input("Enter topk (default 5): ").strip()
    topk = "5"
    if topk_str:
        if not topk_str.isdigit():
             print("Error: topk must be a number.")
             return
        topk = topk_str

    print(f"\nRunning Inference for: '{mention}' with topk={topk}")
    cmd = [sys.executable, "inference.py", "--mention", mention, "--result_encoder_dir", result_encoder_dir, "--topk", topk]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")
    except KeyboardInterrupt:
        print("\nInference interrupted.")

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
        result_encoder_dir = selected_log.get("result_encoder_dir")
        
        # Handle relative paths if necessary (assuming running from root)
        if log_file_path.startswith("./"):
             # It is already relative to root if running from root
             pass
        
        print(f"\nSelected: {selected_log.get('training_log_name')}")
        
        while True:
            print("\nOptions:")
            print("1. get table epochs")
            print("2. Run Eval")
            print("3. Run Inference")
            print("b. Back to log list")
            
            opt = input("Select an option: ").strip()
            
            if opt == '1':
                get_epoch_summaries(log_file_path)
            elif opt == '2':
                run_eval(result_encoder_dir)
            elif opt == '3':
                run_inference(result_encoder_dir)
            elif opt.lower() == 'b':
                break
            else:
                print("Invalid option.")

if __name__ == "__main__":
    main()
