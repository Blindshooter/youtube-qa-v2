import os
import json
import pytube
from pytube import YouTube

def transcribe_whisper(model):
    output = model.transcribe("youtube_video.mp4", verbose=False)

    transcript = output['text']

    return transcript

def generate_subtitles(stream, model, test=False):

    # TRANSCRIPT TAKES A LONG TIME
    if not test:
        transcript = transcribe_whisper(model)
        with open('transcript.txt', 'w') as f:
            f.write(transcript)

    else:
        with open("transcript.txt") as f:
            transcript = f.read()

    return transcript

def get_video_info(link):
    yt_video = YouTube(link)
    v_id = pytube.extract.video_id(link)

    streams = yt_video.streams.filter(only_audio=True)

    stream = streams[0]

    title = yt_video.title

    stream.download(filename="youtube_video.mp4")

    return stream, title
