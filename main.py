import json
import os
from datetime import datetime
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.tools import tool
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAI

import speech_recognition as sr
import subprocess
import time
import gtts
import os
import platform
import pathlib
import google.generativeai as genai
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import cv2
import time
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools
import os
from langchain_community.utilities import OpenWeatherMapAPIWrapper
import getpass
import os
from langchain_google_genai import GoogleGenerativeAI
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import re
import os
from generator import image_generator
import os
import glob






# Define the file path for saving chat history
CHAT_HISTORY_FILE = "chat_history.json"

api_key= "add_api_key"
directory = '/Users/vanshkumarsingh/Desktop/BEEHIVE/pythonProject/generated-pictures'

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

Online = ping_google_dns()
# Set up the LLM which will power our application.

if Online:
    st.toast("Using Gemini.")
    model=GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key)
elif not Online:
    st.toast("Using Mistral")
    model = Ollama(model='mistral:instruct')

chat_history = [] # Store the chat history


# Load chat history from file if exists
if os.path.exists(CHAT_HISTORY_FILE):
    with open(CHAT_HISTORY_FILE, "r") as f:
        try:
            chat_history_data = json.load(f)
            for item in chat_history_data:
                if item['type'] == 'human':
                    chat_history.append(HumanMessage(content=item['content'] ))
                elif item['type'] == 'ai':
                    chat_history.append(AIMessage(content=item['content']))
        except json.JSONDecodeError:
            pass


# Define tools available.

@tool
def get_generated_image_by_rank(rank):
    """Get the image at the specified rank (1-based index) from the generated pictures directory."""
    # Use glob to get all files in the directory
    bar.progress(40)
    files = glob.glob(os.path.join(directory, '*'))
    if not files:
        return "No file found in the directory."

    # Sort files by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    bar.progress(60)
    # Return the file at the specified rank, or the oldest file if out of bounds
    if 1 <= rank <= len(files):
        st.image(files[rank - 1]) # rank is 1-based index
        return "Image at rank {} displayed successfully.".format(rank)
    else:
        st.image(files[-1])  # Return the oldest file if rank is out of bounds
        return "Invalid rank. Displaying the oldest image instead."



@tool
def image_generator_perch(prompt_txt : str) -> str :
    """Generate an image based on the given prompt. The Prompt should be concise"""
    bar.progress(40)
    generator = image_generator(
        base_filename="",
        amount=1,
        prompt= str(prompt_txt),
        prompt_size= 10,
        negative_prompt= "nudity text",
        style="cinematic",
        resolution="512x768",
        guidance_scale=7
    )

    for _ in generator:
        pass
    bar.progress(60)

    files = glob.glob(os.path.join(directory, '*'))
    if not files:
        st.error("No images found in the directory.")
    # Find the newest file based on modification time
    newest_file = max(files, key=os.path.getmtime)
    st.image(newest_file)
    st.snow()
    return str(prompt_txt)


@tool
def get_task_decomposition(placeroute : str , query : str)-> str:
    """This is an RAG it takes an url or an pdf pathname with the query and Get the task decomposition for the given input."""
    bar.progress(30)
    # Set up the LLM which will power our application.
    llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key)

    def differentiate_input(input_string):
        # Regular expression for URL detection
        url_regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
            r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        # Check if the input is a URL
        if re.match(url_regex, input_string):
            return "URL"

        # Check if the input is a file path
        elif os.path.exists(input_string):
            if os.path.isfile(input_string):
                file_extension = os.path.splitext(input_string)[1].lower()
                if file_extension == '.pdf':
                    return "PDF File"
                elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                    return "Image File"
                else:
                    return "Unknown File Type"
            else:
                return "Not a File"

        # If it's neither a URL nor a valid file path
        else:
            return "Invalid Input"

    type = differentiate_input(placeroute)
    bar.progress(40)

    if type  == "URL":
        # Load, chunk and index the contents of the blog.
        loader = WebBaseLoader(web_paths=(placeroute,))
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
    elif type == "PDF File":
        # Generate embeddings for the chunks.
        loader = PyPDFLoader(placeroute)
        splits = loader.load_and_split()
    else:
        return str("No valid input provided")
    bar.progress(50)


    vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="mxbai-embed-large"))

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    bar.progress(60)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    bar.progress(70)
    st.snow()

    return rag_chain.invoke(query)

@tool
def weather(city: str) -> str:
    """Get the current weather for a specified city."""
    bar.progress(40)
    os.environ["OPENWEATHERMAP_API_KEY"] = "94e1c638299ebe20c5b23872f8e58b0d"
    weather = OpenWeatherMapAPIWrapper()
    bar.progress(60)
    return weather.run(city)


@tool
def show_image(image_path:str)->str:
    """
    Display an image given its file path.

    Parameters:
    - image_path (str): The path to the image file.
    """
    bar.progress(40)
    try:
        image = Image.open(image_path)
        st.image(image, caption='Displayed Image')
        return str(f"Image displayed successfully: {image_path}")
    except IOError:
        return str(f"Unable to load image: {image_path}")

@tool
def open_app(app:str)->str:
    """Open the specified application."""
    bar.progress(40)
    subprocess.Popen(['open', '-a', app])
    return f"Opening {app}..."

@tool
def close_app(app_name:str)->str:
    """Close the specified application."""
    bar.progress(40)
    try:
        # Find the process ID (PID) of the application
        pid_cmd = subprocess.Popen(['pgrep', '-i', app_name], stdout=subprocess.PIPE)
        pid_output, _ = pid_cmd.communicate()

        # Convert PID output to string and split if multiple PIDs are found
        pids = pid_output.decode().strip().split()

        for pid in pids:
            # Terminate each process by PID
            subprocess.Popen(['kill', pid])

        return str(f"{app_name} closed successfully.")

    except subprocess.CalledProcessError:
        return str(f"Failed to close {app_name}.")


@tool
def search_duckduckgo(query : str )-> str:
    """Search DuckDuckGo for the given query and return the results."""
    bar.progress(40)
    # Create an instance of the DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()

    # Run the search query
    result = search.run(query)
    bar.progress(60)
    # Return the result
    return result

@tool
def capture_and_display(filename='captured_image.jpg')-> str:
    """Clicks a picture and shows it"""
    bar.progress(40)
    # Initialize the camera
    camera = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)

    # Check if the camera opened successfully
    if not camera.isOpened():
        st.error("Error: Could not open camera.")
        return

    # Give a few seconds for the camera to initialize
    st.info("Capturing image...")
    time.sleep(2)

    # Capture a single frame
    ret, frame = camera.read()
    bar.progress(60)

    if not ret:
        st.error("Error: Failed to capture image.")
        camera.release()
        return

    # Release the camera
    camera.release()

    # Save the captured frame to file
    cv2.imwrite(filename, frame)
    st.balloons()

    # Display the captured image using Streamlit
    st.image(frame, caption='Captured Image')
    return "path = captured_image.jpg"

@tool
def get_image_caption(image_path : str) -> str:
    """
    Generates a short caption for the provided image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A string representing the caption for the image.
    """
    bar.progress(30)
    image = Image.open(image_path).convert('RGB')

    model_name = "Salesforce/blip-image-captioning-large"
    device = "cpu"  # cuda

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    bar.progress(50)

    inputs = processor(image, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=20)
    bar.progress(70)

    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption

@tool
def detect_objects(image_path : str ) -> str:
    """
    Detects objects in the provided image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A string with all the detected objects. Each object as '[x1, x2, y1, y2, class_name, confindence_score]'.
    """
    bar.progress(40)
    image = Image.open(image_path).convert('RGB')

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    bar.progress(60)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        detections += ' {}'.format(model.config.id2label[int(label)])
        detections += ' {}\n'.format(float(score))

    return detections

@tool
def generate_content(prompt : str , image_path : str) -> str:
    """Generate content based on a prompt and an image."""
    bar.progress(30)
    try:
        # Initialize the generative model
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Read image bytes from file
        image_bytes = pathlib.Path(image_path).read_bytes()

        # Prepare input data
        cookie_picture = {
            'mime_type': 'image/png',
            'data': image_bytes
        }
        bar.progress(50)

        # Generate content
        response = model.generate_content([prompt, cookie_picture])

        # Print or return the generated text
        return response.text

    except Exception as e:
        return f"Error occurred: {str(e)}"
@tool
def text_to_speech(text : str , language='en', slow=False , delete_after_playing=True) -> str:

    """
    Convert text to speech and play the audio.

    Parameters:
    - text: The text to convert to speech.
    - language: The language in which you want to convert the text (default is 'en').
    - slow: Whether the speech should be slow (default is False).
    - delete_after_playing: Whether to delete the audio file after playing (default is True).
    """
    bar.progress(40)
    # Create an instance of gTTS
    speech = gtts.gTTS(text=text, lang=language, slow=slow)

    # Save the converted audio to a file
    output_file = "output.mp3"
    speech.save(output_file)

    # Determine the system platform
    system_platform = platform.system()
    bar.progress(60)

    # Play the converted file based on the platform
    if system_platform == "Windows":
        os.system(f"start {output_file}")
    elif system_platform == "Darwin":  # macOS
        os.system(f"afplay {output_file}")
    elif system_platform == "Linux":
        os.system(f"mpg321 {output_file}")  # Ensure mpg321 is installed: sudo apt-get install mpg321
    else:
        return str("Unsupported operating system. Please play the audio file manually.")

    # Optionally, delete the temporary audio file
    if delete_after_playing:
        os.remove(output_file)
    return str(text + " converted to speech and played successfully.")


@tool
def take_screenshot_with_countdown(delay_seconds: int) -> str:
    """Take a screenshot after a specified delay with a countdown."""
    bar.progress(40)
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
        bar.progress(60)


        # Get the current user's home directory
        home_dir = "/Users/vanshkumarsingh"  # Replace with your actual home directory path

        # Define the filename and path for the screenshot
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        screenshot_path = f"{home_dir}/Desktop/screenshot_{timestamp}.png"

        # Construct the shell command to capture the screenshot
        command = ['screencapture', '-x', screenshot_path]  # -x flag to not play sounds

        # Execute the command using subprocess
        subprocess.run(command, check=True)

        return str(f"Screenshot saved: {screenshot_path}")

    except subprocess.CalledProcessError as e:
        return str(f"Error: {e}")


@tool
def set_timer(minutes : float) -> str:
    """Set a timer for the specified number of minutes."""
    bar.progress(40)
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
        bar.progress(60)

        # Execute the AppleScript command using osascript
        subprocess.run(['osascript', '-e', applescript], check=True)

        return str(f"Timer set for {minutes} minutes.")

    except subprocess.CalledProcessError as e:
        return str(f"Error setting timer: {e}")


@tool
def run_command(command: str) -> str:
    """Run a shell command. Input should be a valid shell command. Return the output of the command."""
    bar.progress(40)
    try:
        # Run the command
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        bar.progress(60)

        # Construct the output string
        output_str = f"Output:\n{result.stdout}"
        if result.stderr:
            output_str += f"\nALT Output (if any):\n{result.stderr}"

        return output_str

    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e}"


@tool
def recognize_speech_from_microphone(input: str) -> str:
    """Recognize speech from microphone using Google Web Speech API."""
    bar.progress(40)
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Capture audio from microphone
    with sr.Microphone() as source:
        print("Listening...")

        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)

        # Listen for user input
        audio = recognizer.listen(source)
        bar.progress(60)

        try:
            print("Recognizing...")
            # Using Google Web Speech API to recognize audio
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return str(text)
        except sr.UnknownValueError:
            return str("Google Web Speech API could not understand audio")

        except sr.RequestError as e:
            return str(f"Could not request results from Google Web Speech API; {e}")


@tool
def repl(input: str) -> str:
    """A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."""
    bar.progress(40)
    python_repl = PythonREPL()
    return python_repl.run(input)

@tool
def converse(input: str) -> str:
    """Provide a natural language response using the user input."""
    bar.progress(40)
    return model.invoke(input)

#tools = [repl, converse ,recognize_speech_from_microphone , ]
tools = [
    get_generated_image_by_rank,
    image_generator_perch,
    get_task_decomposition,
    weather,
    show_image,
    open_app,
    close_app,
    search_duckduckgo,
    capture_and_display,
    get_image_caption,
    detect_objects,
    generate_content,
    text_to_speech,
    take_screenshot_with_countdown,
    set_timer,
    run_command,
    recognize_speech_from_microphone,
    repl,
    converse
]


# Configure the system prompts
rendered_tools = render_text_description(tools)

system_prompt = f"""You answer questions with simple answers and no funny stuff , You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

{rendered_tools}

Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys. The value associated with the 'arguments' key should be a dictionary of parameters."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
     MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ]
)

# Define a function which returns the chosen tools as a runnable, based on user input.
def tool_chain(model_output):
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool

# The main chain: an LLM with tools.
chain = prompt | model | JsonOutputParser() | tool_chain



def save_chat_history():
    chat_history_data = []
    for message in chat_history:
        if isinstance(message, HumanMessage) or isinstance(message, AIMessage):
            chat_history_data.append({"type": "human" if isinstance(message, HumanMessage) else "ai",
                                      "content": message.content,
                                      })

    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(chat_history_data, f, default=str)  # Use default=str to serialize datetime if needed

def clear_chat_history():
    global chat_history
    chat_history = []
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)

def delete_all_files():
    # Get all files in the directory
    files = glob.glob(os.path.join(directory, '*'))
    for file in files:
        os.remove(file)
    return len(files)




# Set up message history.
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("From calculations to image generation, data analysis to task prioritization, I'm here to assist. Always on, always learning. How can I help you today?")

# Set the page title.
st.title("Ascendant Ai")

# Render the chat history.
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# React to user input
if input := st.chat_input("What is up?"):

    if input == "/clear":
        clear_chat_history()
        #print("Chat history cleared.")
        st.chat_message("assistant").write("Chat history cleared.")
        delete_all_files()
        st.toast("Data Cleared")

    else:
        # Display user input and save to message history.
        st.chat_message("user").write(input)
        msgs.add_user_message(input)

        # Invoke chain to get response.
        bar = st.progress(0)
        response = chain.invoke({"input": input, "chat_history": chat_history})
        chat_history.append(HumanMessage(content=input))
        chat_history.append(AIMessage(content=response))
        bar.progress(90)

        # Display AI assistant response and save to message history.
        st.chat_message("assistant").write(str(response))
        msgs.add_ai_message(response)

        save_chat_history()
        st.toast("Context Updated")
        bar.progress(100)

        # Ensure the model retains context
        #msgs.add_ai_message(model.invoke(input))

