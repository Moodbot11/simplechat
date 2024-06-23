import os
import base64
import streamlit as st
import openai
import tempfile
import logging
from pydub import AudioSegment
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np
import queue
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client instance
def init_openai(api_key):
    openai.api_key = api_key
    return openai

# Function to generate speech from text
def generate_speech(text, client):
    response = client.Audio.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    audio_content = response["audio"]
    st.audio(audio_content, format="audio/mp3")

# Function to transcribe speech to text
def transcribe_speech(file_buffer, client):
    response = client.Audio.transcriptions.create(
        file=file_buffer,
        model="whisper-1"
    )
    return response["text"]

# Audio processor for webrtc_streamer
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_queue = queue.Queue()

    def recv_queued(self, frames: av.AudioFrame) -> av.AudioFrame:
        for frame in frames:
            audio = frame.to_ndarray()
            audio = (audio * 32767).astype(np.int16)  # Convert to 16-bit PCM
            self.audio_queue.put(audio.tobytes())
        return frames[-1]

# Read OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the OpenAI API key is provided
if not openai_api_key:
    st.error("Please set the OpenAI API key as an environment variable.")
    st.stop()

# Initialize OpenAI client
openai_client = init_openai(openai_api_key)

# Main title and caption
st.title("💬 Chatbot with TTS and STT")
st.caption("🚀 A Streamlit chatbot powered by OpenAI with TTS and STT")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display message history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle voice input
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "audio": True,
        "video": False,
    },
    audio_processor_factory=AudioProcessor,
)

if webrtc_ctx.audio_processor:
    audio_processor = webrtc_ctx.audio_processor
    audio_data = b"".join(list(audio_processor.audio_queue.queue))

    if audio_data:
        try:
            logger.info(f"{datetime.now()} - Audio data captured.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                temp_audio_file.write(audio_data)
                temp_audio_file_path = temp_audio_file.name

            with open(temp_audio_file_path, "rb") as audio_file:
                prompt = transcribe_speech(audio_file, openai_client)
                st.write(f"Transcription: {prompt}")

                st.session_state["messages"].append({"role": "user", "content": prompt})

                logger.info(f"{datetime.now()} - Sending transcription to OpenAI ChatCompletion API...")
                response = openai_client.ChatCompletion.create(
                    model="gpt-4",
                    messages=st.session_state["messages"]
                )
                msg = response.choices[0]["message"]["content"]
                st.session_state["messages"].append({"role": "assistant", "content": msg})
                st.write(f"Assistant: {msg}")

                # Convert response to audio and play it
                logger.info(f"{datetime.now()} - Converting response to audio...")
                generate_speech(msg, openai_client)
        except Exception as e:
            logger.error(f"{datetime.now()} - An error occurred: {e}")
            st.error(f"An error occurred: {e}")

# Provide options to generate speech or upload voice for transcription
if prompt := st.chat_input():
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = openai_client.ChatCompletion.create(
        model="gpt-4",
        messages=st.session_state["messages"]
    )

    msg = response.choices[0]["message"]["content"]
    st.session_state["messages"].append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

    if st.button("Generate Speech"):
        generate_speech(msg, openai_client)

uploaded_file = st.file_uploader("Upload Audio for Transcription", type=["mp3", "wav"])
if uploaded_file is not None:
    file_buffer = BytesIO(uploaded_file.read())
    transcribed_text = transcribe_speech(file_buffer, openai_client)
    st.write("Transcribed text: ", transcribed_text)

