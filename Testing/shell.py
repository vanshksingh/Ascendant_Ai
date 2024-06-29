import subprocess


def run_command(command):
    try:
        # Run the command
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        # Print the output
        print("Output:\n", result.stdout)
        print("Error (if any):\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error executing command:", e)


if __name__ == "__main__":
    # Example command to run
    command = "python3 --version"

    # Run the command
    run_command(command)
