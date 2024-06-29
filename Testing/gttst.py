import gtts
import os
import platform

def text_to_speech(text, language='en', slow=False, delete_after_playing=True):
    """
    Convert text to speech and play the audio.

    Parameters:
    - text: The text to convert to speech.
    - language: The language in which you want to convert the text (default is 'en').
    - slow: Whether the speech should be slow (default is False).
    - delete_after_playing: Whether to delete the audio file after playing (default is True).
    """
    # Create an instance of gTTS
    speech = gtts.gTTS(text=text, lang=language, slow=slow)

    # Save the converted audio to a file
    output_file = "output.mp3"
    speech.save(output_file)

    # Determine the system platform
    system_platform = platform.system()

    # Play the converted file based on the platform
    if system_platform == "Windows":
        os.system(f"start {output_file}")
    elif system_platform == "Darwin":  # macOS
        os.system(f"afplay {output_file}")
    elif system_platform == "Linux":
        os.system(f"mpg321 {output_file}")  # Ensure mpg321 is installed: sudo apt-get install mpg321
    else:
        print("Unsupported operating system. Please play the audio file manually.")

    # Optionally, delete the temporary audio file
    if delete_after_playing:
        os.remove(output_file)

# Example usage
text_to_speech("Hello, welcome to the world of text-to-speech conversion using gTTS in Python!")
