import speech_recognition as sr

def recognize_speech_from_microphone():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Capture audio from microphone
    with sr.Microphone() as source:
        print("Listening...")

        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)

        # Listen for user input
        audio = recognizer.listen(source)

        try:
            print("Recognizing...")
            # Using Google Web Speech API to recognize audio
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")
            return None

# Example usage
if __name__ == "__main__":
    recognize_speech_from_microphone()
