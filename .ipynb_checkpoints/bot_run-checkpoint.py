import sys
import os
import subprocess
import argparse

def send_to_keybase(message):
    """Send a message to Keybase."""
    os.system(f'keybase chat send comunelab.sandbox --channel lucallegri_instances "{message}"')

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run a script and forward its outputs to Keybase.")
    parser.add_argument("--script", required=True, help="Path to the script to run.")
    args = parser.parse_args()

    script_path = args.script
    if not os.path.isfile(script_path):
        print(f"Error: Script '{script_path}' not found.")
        sys.exit(1)

    send_to_keybase(f"Running script '{script_path}'")

    # Run the target script and capture its output
    process = subprocess.Popen(
        [sys.executable, "-u", script_path],  # Use the same Python interpreter to run the target script
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # Capture and handle stdout and stderr in real-time
        for line in process.stdout:
            if line.strip() != "":
                print(line)  # Print to the console
                send_to_keybase(line)

        for line in process.stderr:
            if line.strip() != "":
                print(line, file=sys.stderr)  # Print to the console as an error
                send_to_keybase(f"[ERROR] {line}")

        # Wait for the process to complete
        process.wait()

        if process.returncode != 0:
            send_to_keybase(f"Script '{script_path}' exited with code {process.returncode}")
        else:
            send_to_keybase(f"Script '{script_path}' ended correctly.")
    except Exception as e:
        send_to_keybase(f"Wrapper script failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
