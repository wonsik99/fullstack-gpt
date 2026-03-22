import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import os
import glob
from openai import OpenAI
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.storage import LocalFileStore
from langchain_core.embeddings import CacheBackedEmbeddings

llm = ChatOpenAI(
    model_name="gpt-5.4-nano",
    temperature=0.1
    )

@st.cache_resource
def embed_file(file_path):

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever() #k=4 is default
    return retriever

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
    status.update(state="complete")

    transcript_tab, summary_tab, qa_tab = st.tabs(["Transcript", "Summary", "Q&A"])
    
    with transcript_tab:
        with open(transcript_path, "r") as f:
            st.write(f.read())
    
    with summary_tab:
        start = st.button("Generate Summary")
        
        if start:
            loader = TextLoader(transcript_path)
            
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800,
                chunk_overlap=100
            )
            docs = loader.load_and_split(text_splitter=splitter)
            
            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:
                """
            )
            first_summary_chain = first_summary_prompt | llm
            
            summary = first_summary_chain.invoke({
                "text": docs[0].page_content
            }).content
            
            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_answer}
                We have the opportunity to refine the existing summary (only if needed) with some more context below
                ------
                {text}
                ------
                Given the new context, refine the original summary.
                If the context is not useful, RETURN the original summary.
                """
            )
            refine_chain = refine_prompt | llm
            
            with st.status("Refining Summary...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Refining Summary... ({i+1}/{len(docs)-1})", state="running")
                    summary = refine_chain.invoke({
                        "existing_answer": summary,
                        "text": doc.page_content
                    }).content
            status.update(state="complete")
            st.write(summary)

    with qa_tab:
        retriever = embed_file(transcript_path)
        
        response = retriever.invoke("Who is the main speaker")

        st.write(response)

        # TODO: Add chat interface
        # Map rerank chain or Map reduce chain

