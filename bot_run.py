# Script to get status of running code via Keybase.
# Usage: '(bash) python3 bot_run.py --script your_script --channel your_comunelab.sandbox_channel'
# All the prints of your_script are captured and sent real time to your_comunelab.sandbox_channel. If you don't want all
# the prints to be sent you can only send the prints that contains specific words using the flag --words.
# For example if your_script prints "Simulation started" when it starts and "Simulation ended" when it ends, you can run
# '(bash) python3 bot_run.py --script your_script --channel your_comunelab.sandbox_channel --words started ended'.

import os
import sys
import json
import signal
import argparse
import datetime
import threading
import subprocess

global last_line
last_line = ""  # Initialize last_line
last_line_lock = threading.Lock()  # Initialize the Lock
stop_event = threading.Event()  # Event to signal main process to stop

def send_to_keybase(message, channel):
    """Send a message to Keybase."""
    send_to = f'comunelab.sandbox --channel {channel}'
    command = f'keybase chat send {send_to} "{message}"'
    result = os.system(command)
    if result != 0:
        print(f"[ERROR] Failed to send message to Keybase: {message}")

def should_send_message(line, words):
    """Determine if a message should be sent based on the provided words."""
    if words:
        return any(word in line for word in words)
    return True

def monitor_keybase(channel, trigger_message):
    """Monitor Keybase for specific incoming messages."""
    command = f'keybase chat api-listen'
    try:
        with subprocess.Popen(command.split(), stdout=subprocess.PIPE, text=True) as proc:
            for line in proc.stdout:
                try:
                    message = json.loads(line)
                    content = message.get("msg", {}).get("content", {})
                    if content.get("type") == "text":
                        text = content["text"]["body"]
                        sender = message["msg"]["sender"]["username"]
                        channel_name = message["msg"]["channel"]["topic_name"]

                        if channel_name == channel and text == trigger_message:
                            print(f"[INFO] Trigger message received from {sender}: {text}")
                            respond_to_trigger(channel)
                except json.JSONDecodeError:
                    print(f"[ERROR] Failed to parse JSON: {line.strip()}")
    except Exception as e:
        print(f"[ERROR] Keybase monitoring failed: {e}")

def respond_to_trigger(channel):
    """Callback function when a trigger message is received."""
    with last_line_lock:  # Safely access last_line
        message = last_line or "[INFO] No output captured yet."
    send_to_keybase(message, channel)  # Adjust the channel name if needed

def handle_interrupt(signal, frame):
    """Handle the interrupt signal (Ctrl+C)."""
    print("\n[INFO] Script interrupted. Cleaning up...")
    if process.poll() is None:
        process.terminate()
        print("[INFO] Subprocess terminated.")
    sys.exit(0)

def run_script(script_path, channel, words):
    """Run the specified script and forward its output to Keybase."""
    send_to_keybase(f"[INFO] Starting script '{script_path}' at {datetime.datetime.now()}.", channel)

    try:
        global process  # Make process accessible in the signal handler
        executable = sys.executable
        process = subprocess.Popen(
            [executable, "-u", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        for line in iter(process.stdout.readline, ""):
            if line.strip():
                print(line, end="")
                with last_line_lock:  # Safely update last_line
                    global last_line
                    last_line = line.strip()
                if should_send_message(line, words):
                    send_to_keybase(f"[INFO] {line.strip()}", channel)

        for line in iter(process.stderr.readline, ""):
            if line.strip():
                print(line, file=sys.stderr, end="")
                send_to_keybase(f"[ERROR] {line.strip()}", channel)

        process.wait()  # Wait for the process to complete

        if process.returncode == 0:
            send_to_keybase(f"[INFO] Script '{script_path}' completed successfully at {datetime.datetime.now()}.", channel)
        else:
            send_to_keybase(f"[ERROR] Script '{script_path}' exited with code {process.returncode}.", channel)
    except Exception as e:
        send_to_keybase(f"[ERROR] Failed to run script '{script_path}': {e}", channel)
    finally:
        if process.poll() is None:  # If process is still running, terminate it
            process.terminate()
            print("[INFO] Subprocess terminated.")
        stop_event.set()  # Signal to stop when the subprocess ends

def main():
    parser = argparse.ArgumentParser(description="Run a script and monitor Keybase for messages.")
    parser.add_argument("--script", required=True, help="Path to the script to run.")
    parser.add_argument("--channel", required=True, help="Channel to monitor.")
    parser.add_argument("--words", nargs="*", help="Specific words to filter messages.")
    args = parser.parse_args()

    script_path = args.script
    channel = args.channel
    words = args.words

    if not os.path.isfile(script_path):
        print(f"Error: Script '{script_path}' not found.")
        sys.exit(1)

    print(f"[INFO] Monitoring Keybase channel '{channel}'.")
    threading.Thread(target=monitor_keybase, args=(channel, 'Status?', respond_to_trigger), daemon=True).start()

    signal.signal(signal.SIGINT, handle_interrupt)
    print(f"[INFO] Running script '{script_path}'. Press Ctrl+C to stop.")
    threading.Thread(target=run_script, args=(script_path, channel, words), daemon=True).start()

    try:
        while not stop_event.is_set():  # Wait until the stop event is set
            pass
    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")

if __name__ == "__main__":
    main()
