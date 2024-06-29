import subprocess


def ping_google_dns():
    try:
        # Run the ping command
        output = subprocess.run(['ping', '8.8.8.8', '-c', '1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check the return code
        if output.returncode == 0:
            return True
        else:
            return False
    except Exception as e:
        # In case of any exceptions, return False
        print(f"An error occurred: {e}")
        return False


# Example usage:
if __name__ == "__main__":
    result = ping_google_dns()
    print(f"Ping successful: {result}")
