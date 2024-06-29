import subprocess
import time


def take_screenshot_with_countdown(delay_seconds=5):
    try:
        # Display countdown using AppleScript
        countdown_script = f'''
            set initialTime to current date
            set endTime to initialTime + {delay_seconds}

            repeat with remainingTime from {delay_seconds} to 0 by -1
                display notification "Screenshot in " & remainingTime & " seconds" with title "Screenshot Countdown"
                delay 1
            end repeat

            display notification "Taking screenshot now..." with title "Screenshot Countdown" sound name "Glass"
        '''

        # Execute AppleScript countdown
        subprocess.run(['osascript', '-e', countdown_script], check=True)

        # Wait for the specified delay
        print(f"Waiting for {delay_seconds} seconds before taking screenshot...")
        time.sleep(delay_seconds)

        # Get the current user's home directory
        home_dir = "/Users/vanshkumarsingh"  # Replace with your actual home directory path

        # Define the filename and path for the screenshot
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        screenshot_path = f"{home_dir}/Desktop/screenshot_{timestamp}.png"

        # Construct the shell command to capture the screenshot
        command = ['screencapture', '-x', screenshot_path]  # -x flag to not play sounds

        # Execute the command using subprocess
        subprocess.run(command, check=True)

        print(f"Screenshot saved: {screenshot_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


# Example usage: Take a screenshot after a 5-second delay with countdown
if __name__ == "__main__":
    take_screenshot_with_countdown(5)
