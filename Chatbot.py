import os
import base64
import streamlit as st
import openai
import tempfile
import logging
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import queue
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to convert text to speech using OpenAI's API and return audio bytes
def convert_text_to_speech(text, api_key, model="tts-1", voice="alloy"):
    openai.api_key = api_key
    logger.info(f"{datetime.now()} - Converting text to speech")
    response = openai.Audio.create(
        model=model,
        voice=voice,
        input=text
    )
    audio_content = response['audio']
    return audio_content

# Function to convert speech to text using OpenAI's API and return text transcription
def convert_speech_to_text(audio_data, api_key):
    openai.api_key = api_key
    logger.info(f"{datetime.now()} - Converting speech to text")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file_path = temp_audio_file.name
    with open(temp_audio_file_path, "rb") as audio_file:
        response = openai.Audio.transcriptions.create(
            file=audio_file,
            model="whisper-1"
        )
    return response['text']

# Function to convert audio bytes to base64
def audio_bytes_to_base64(audio_bytes, audio_format="mp3"):
    logger.info(f"{datetime.now()} - Converting audio bytes to base64")
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    audio_str = f"data:audio/{audio_format};base64,{audio_base64}"
    return audio_str

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

# Main title and caption
st.title("💬 Chatbot with TTS and STT")
st.caption("🚀 A Streamlit chatbot powered by OpenAI with TTS and STT")

# Initialize messages in session state if not already present
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Handle voice input
webrtc_ctx = webrtc_streamer(
    key="audio",
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
            prompt = convert_speech_to_text(audio_data, openai_api_key)
            st.write(f"Transcription: {prompt}")

            openai.api_key = openai_api_key
            st.session_state.messages.append({"role": "user", "content": prompt})

            logger.info(f"{datetime.now()} - Sending transcription to OpenAI ChatCompletion API...")
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=st.session_state["messages"]
            )
            msg = response.choices[0].message['content']
            st.session_state["messages"].append({"role": "assistant", "content": msg})
            st.write(f"Assistant: {msg}")

            logger.info(f"{datetime.now()} - Converting response to audio...")
            audio_content = convert_text_to_speech(msg, openai_api_key)
            audio_str = audio_bytes_to_base64(audio_content)
            st.audio(audio_str, format="audio/mp3")
        except Exception as e:
            logger.error(f"{datetime.now()} - An error occurred: {e}")
            st.error(f"An error occurred: {e}")

# Display chat history
for msg in st.session_state["messages"]:
    st.write(f"{msg['role'].capitalize()}: {msg['content']}")
