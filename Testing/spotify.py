import urllib.request
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import ssl
# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id='127e268d9c6a4885a675a662aba16e23', client_secret='2b2e9a9033544c24b735a83df49702fd')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Example track ID
track_id = '7qiZfU4dY1lWllzX7mPBI3'

# Get track info
track_info = sp.track(track_id)
print(f"Found track: {track_info['name']} by {track_info['artists'][0]['name']}")
print(f"Track info: {track_info}")

# Get album cover URL
album_cover_url = track_info['album']['images'][0]['url']  # assuming the first image is the largest

# Function to display album cover
def display_album_cover(url):
    # Download the image
    urllib.request.urlretrieve(url, "album_cover.jpg")
    print("Album cover saved as album_cover.jpg")

# Display the album cover
display_album_cover(album_cover_url)
