import subprocess

# Replace 'Mail' with 'Mail.app' if needed
def open_app(app):
    subprocess.Popen(['open', '-a', app])




def close_app(app_name):
    try:
        # Find the process ID (PID) of the application
        pid_cmd = subprocess.Popen(['pgrep', '-i', app_name], stdout=subprocess.PIPE)
        pid_output, _ = pid_cmd.communicate()

        # Convert PID output to string and split if multiple PIDs are found
        pids = pid_output.decode().strip().split()

        for pid in pids:
            # Terminate each process by PID
            subprocess.Popen(['kill', pid])

        print(f"{app_name} closed successfully.")

    except subprocess.CalledProcessError:
        print(f"Failed to close {app_name}.")


# Example usage:
close_app('Mail')
