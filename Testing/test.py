import os
import glob


def get_newest_file(directory):
    # Use glob to get all files in the directory
    files = glob.glob(os.path.join(directory, '*'))
    if not files:
        return None

    # Find the newest file based on modification time
    newest_file = max(files, key=os.path.getmtime)
    return newest_file


# Specify the directory
directory = '/Users/vanshkumarsingh/Desktop/BEEHIVE/pythonProject/generated-pictures'

# Get the newest file
newest_file = get_newest_file(directory)
print(f"The newest file is: {newest_file}")

# If you want to display the image using Streamlit
if newest_file:
    import streamlit as st
    from PIL import Image

    image = Image.open(newest_file)
    st.image(image, caption="Newest Image")
else:
    print("No files found in the directory.")
