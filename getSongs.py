'''
Author: Kyle Vu
Date: 7/19/2018
Purpose: Get Spotify playlist into data frame format
'''
# load necessary libraries
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas
from itertools import chain
import csv

def main():

    client_id = input("Enter your client id:")
    secret = input("Enter your client secret:")
    cred = SpotifyClientCredentials(client_id = client_id, client_secret = secret)
    spot = spotipy.Spotify(client_credentials_manager = cred) # create API object

    my_name = input("Enter your Spotify username:")
    like_id = input("Enter the ID of the playlist containing songs you like:")
    dislike_id = input("Enter the ID of the playlist containing songs you dislike:")

    like_playlist = get_playlist_tracks(my_name, like_id)
    like_frame = make_song_frame(like_playlist)
    like_frame.to_csv("spotifyLike.csv")

    dislike_playlist = get_playlist_tracks(my_name, dislike_id)
    dislike_frame = make_song_frame(dislike_playlist)
    dislike_frame.to_csv("spotifyDislike.csv")

main()

# create function to work around 100-song limit
# take in Spotify username and playlist id, return playlist dictionary
def get_playlist_tracks(name, playlist_id):
    results = spot.user_playlist_tracks(name, playlist_id)
    playlist = results['items']
    while results['next']:
        results = spot.next(results)
        playlist.extend(results['items'])
    return playlist

# take dictionary object, return playlist as data frame
def make_song_frame(playlist):
    id_lst = []
    title_lst = []
    artist_lst = []
    for i in range(len(playlist)):
        id_lst.append(playlist[i]['track']['id'])
        title_lst.append(playlist[i]['track']['name'])
        artist_lst.append(playlist[i]['track']['artists'][0]['name'])

    features_lst = []
    for song_id in id_lst:
        audio_features = spot.audio_features(song_id)
        features_lst.append(audio_features)

    features_lst = list(chain.from_iterable(features_lst)) # unnest features
    title_frame = pandas.DataFrame(title_lst)
    artist_frame = pandas.DataFrame(artist_lst)
    frivolity_frame = pandas.concat([title_frame, artist_frame], axis = 1)
    feature_frame = pandas.DataFrame(features_lst)
    song_frame = pandas.concat([frivolity_frame, feature_frame], axis = 1)

    return song_frame
