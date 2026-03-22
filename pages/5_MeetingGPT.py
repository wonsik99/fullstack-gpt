import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import os
import glob
from openai import OpenAI
import httpx

@st.cache_data()
def extract_audio_from_video(video_path, audio_path):
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)

@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder, transcript_path):
    if os.path.exists(transcript_path):
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    whole = math.ceil(len(track) / chunk_len)

    for i in range(whole):
        chunk = track[i * chunk_len:(i + 1) * chunk_len]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3", format="mp3")

@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if os.path.exists(destination):
        return
    client = OpenAI(http_client=httpx.Client(http2=False))
    files = glob.glob(f"{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                timeout=600
            )
            text_file.write(transcript.text)

st.set_page_config(page_title="MeetingGPT", page_icon="💼")

st.title("MeetingGPT")

st.markdown(
    """ 
    Welcome to MeetingGPT!
    Upload a vidoe and I will give you a transcript,
    a summary and a chat bot to ask any questions about it.
                
    Get started by uploading a video file in the sidebar.
    """
)

with st.sidebar:
    video = st.file_uploader("Video", type = ["mp4", "avi", "mov", "mkv", "webm"])

if video:
    chunks_folder = ".cache/chunks"
    
    with st.status("Loading Video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = f"{'.'.join(video_path.split('.')[:-1])}.mp3"
        transcript_path = f"{'.'.join(video_path.split('.')[:-1])}.txt"

        with open(video_path, "wb") as f:
            f.write(video_content)

    status.update(label="Extracting Audio...", state="running")
    extract_audio_from_video(video_path, audio_path)
    
    status.update(label="Cutting Audio...", state="running")
    cut_audio_in_chunks(audio_path, 10, chunks_folder, transcript_path)

    status.update(label="Transcribing Audio...", state="running")
    transcribe_chunks(chunks_folder, transcript_path)

transcript_tab, summary_tab, qa_tab = st.tabs(["Transcript", "Summary", "Q&A"])

with transcript_tab:
    with open(transcript_path, "r") as f:
        st.write(f.read())

with summary_tab:
    with open(transcript_path, "r") as f:
        transcript = f.read()
        
