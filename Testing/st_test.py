import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import time




import streamlit as st
from playwright.sync_api import sync_playwright
import requests
import re
from random import random as uniform

# Setup logging in Streamlit
st.set_page_config(page_title="AI Text to Image Generator Key Finder")
st.title("AI Text to Image Generator Key Finder")


import streamlit as st
import os

def save_uploaded_file(uploadedfile):
    with open(os.path.join("uploads", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.join("uploads", uploadedfile.name)

st.title("Document Uploader")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    st.write("File saved at:", file_path)


def log_info(message):
    st.write(message)

def get_url_data():
    url_data = []
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()

        def request_handler(request):
            request_info = f'{request.method} {request.url}'
            log_info(request_info)
            url_data.append(request.url)

        page.on("request", request_handler)  # capture traffic

        page.goto('https://perchance.org/ai-text-to-image-generator')

        iframe_element = page.query_selector('xpath=//iframe[@src]')
        frame = iframe_element.content_frame()
        frame.click('xpath=//button[@id="generateButtonEl"]')

        key = None
        while key is None:
            pattern = r'userKey=([a-f\d]{64})'
            all_urls = ''.join(url_data)
            keys = re.findall(pattern, all_urls)
            if keys:
                key = keys[0]
            url_data = []

            page.wait_for_timeout(1000)

        browser.close()

    return key

def get_key():
    """
    1. verify key in last_key.txt if there is one
    2. if not, find one through sending a request
    """

    key = None
    try:
        with open('last-key.txt', 'r') as file:
            line = file.readline().strip()

        if line != '':
            verification_url = 'https://image-generation.perchance.org/api/checkVerificationStatus'
            user_key = line
            cache_bust = uniform()
            verification_params = {
                'userKey': user_key,
                '__cacheBust': cache_bust
            }

            response = requests.get(verification_url, params=verification_params)
            if 'not_verified' not in response.text:
                key = line

        if key is not None:
            log_info(f'Found working key {key[:10]}... in file.')
            return key

        log_info(f'Key no longer valid. Looking for a new key...')
        key = get_url_data()

        log_info(f'Found key {key[:10]}...')
        with open('last-key.txt', 'w') as file:
            file.write(key)

    except Exception as e:
        log_info(f'Error: {str(e)}')

    return key


# Main Streamlit app

st.title('Interactive Graph Creation')

# Example data
simple_data = pd.DataFrame({
    'Date': pd.date_range('2023-01-01', periods=10),
    'Value': np.random.randn(10)
})

# Simulating more complex data
np.random.seed(0)
n = 50
random_x = np.random.randn(n)
random_y = np.random.randn(n)
random_size = np.random.randn(n)
random_category = np.random.choice(['A', 'B', 'C'], n)
complex_data = pd.DataFrame({
    'X': random_x,
    'Y': random_y,
    'Size': random_size,
    'Category': random_category
})

# UI elements for user interaction
st.sidebar.header('User Interaction')

# Dropdown for selecting graph type
graph_type = st.sidebar.selectbox('Select Graph Type:', ['Line', 'Scatter', 'Bar'])

# Radio buttons for choosing dataset
data_choice = st.sidebar.radio('Choose Data:', ['Simple Data', 'Complex Data'])

# Conditional selection of data based on user choice
if data_choice == 'Simple Data':
    selected_data = simple_data
else:
    selected_data = complex_data

# Slider for setting number of data points to display
num_points = st.sidebar.slider('Number of Data Points:', min_value=5, max_value=len(selected_data), value=10, step=5)

# Checkbox for displaying grid lines
show_grid = st.sidebar.checkbox('Show Grid Lines', value=True)

# Text input for setting graph title
graph_title = st.sidebar.text_input('Graph Title:', 'Graph')

# Button to create graph
if st.sidebar.button('Create Graph'):
    st.sidebar.success('Graph Created!')


# Chat-like interface for additional commands
st.header('Chat Interface')
user_input = st.text_input('Enter command:')
if user_input:
    st.info(f'Command received: "{user_input}"')

# Show a spinner during a process
with st.spinner(text="In progress"):
    time.sleep(3)
    st.success("Done")

# Show and update progress bar
bar = st.progress(50)
time.sleep(3)
bar.progress(100)

with st.status("Authenticating...") as s:
    time.sleep(2)
    st.write("Some long response.")
    s.update(label="Response")

st.balloons()
st.snow()
st.toast("Warming up...")
st.error("Error message")
st.warning("Warning message")
st.info("Info message")
st.success("Success message")
st.exception("lol")

import os
import glob
import streamlit as st
from PIL import Image
import io


def get_newest_file(directory):
    # Use glob to get all files in the directory
    files = glob.glob(os.path.join(directory, '*'))
    if not files:
        return None

    # Find the newest file based on modification time
    newest_file = max(files, key=os.path.getmtime)
    return newest_file


def delete_all_files(directory):
    # Get all files in the directory
    files = glob.glob(os.path.join(directory, '*'))
    for file in files:
        os.remove(file)
    return len(files)






# Specify the directory
directory = '/Users/vanshkumarsingh/Desktop/BEEHIVE/pythonProject/generated-pictures'

# Get the newest file
newest_file = get_newest_file(directory)

if newest_file:
    # Open the image
    image = Image.open(newest_file)

    # Display the image in Streamlit
    st.image(image, caption="Newest Image")

    # Convert image to bytes for download
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr = img_byte_arr.getvalue()

    # Add download button
    st.download_button(
        label="Download Image",
        data=img_byte_arr,
        file_name=os.path.basename(newest_file),
        mime=f"image/{image.format.lower()}"
    )
else:
    st.write("No files found in the directory.")

# Add delete all files button
if st.button("Delete All Files"):
    num_deleted_files = delete_all_files(directory)
    st.write(f"Deleted {num_deleted_files} files from the directory.")
