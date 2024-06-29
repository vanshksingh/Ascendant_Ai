import pathlib

import google.generativeai as genai

genai.configure(api_key="AIzaSyAIsE4C0ZjwCuO0A6S7IEjszpY9MBjAgWE")


model = genai.GenerativeModel('gemini-1.5-flash')

cookie_picture = {
    'mime_type': 'image/png',
    'data': pathlib.Path('/Users/vanshkumarsingh/Desktop/test5.png').read_bytes()
}
prompt = "What do you see ?"

response = model.generate_content(

    [prompt, cookie_picture]
)
print(response.text)