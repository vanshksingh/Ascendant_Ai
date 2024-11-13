import streamlit as st
from playwright.sync_api import sync_playwright
import requests
import re
from random import random as uniform

# Setup logging in Streamlit



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

# Main code to run in Streamlit