import subprocess


def set_timer(minutes):
    try:
        # Convert minutes to seconds
        timer_duration_seconds = minutes * 60

        # Construct the AppleScript command
        applescript = f'''
            set initialTime to current date
            set endTime to initialTime + {timer_duration_seconds}

            repeat with remainingTime from {timer_duration_seconds} to 0 by -1
                set remainingMinutes to remainingTime div 60
                set remainingSeconds to remainingTime mod 60

                display notification "Timer: " & remainingMinutes & "m " & remainingSeconds & "s" with title "Timer" 
                delay 1

                
            end repeat

            # Play a final sound after timer completes (adjust the sound path if needed)
            repeat 10 times
                do shell script "afplay /System/Library/Sounds/Glass.aiff"
                delay 1
            end repeat

            display notification "Timer done!" with title "Timer"
        '''

        # Execute the AppleScript command using osascript
        subprocess.run(['osascript', '-e', applescript], check=True)

        print(f"Timer set for {minutes} minutes.")

    except subprocess.CalledProcessError as e:
        print(f"Error setting timer: {e}")


# Example usage: Set a timer for 5 minutes
if __name__ == "__main__":
    set_timer(.1)
