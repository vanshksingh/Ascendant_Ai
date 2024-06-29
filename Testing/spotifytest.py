import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
scope = "user-library-read"



# Initialize Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id='127e268d9c6a4885a675a662aba16e23', client_secret='2b2e9a9033544c24b735a83df49702fd')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


results = sp.current_user_saved_tracks()
for idx, item in enumerate(results['items']):
    track = item['track']
    print(idx, track['artists'][0]['name'], " – ", track['name'])