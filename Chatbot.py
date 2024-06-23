import os
import base64
import streamlit as st
import openai
import tempfile
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np
import queue

# Function to convert text to speech using OpenAI's API and return audio bytes
def convert_text_to_speech(text, api_key, model="tts-1", voice="alloy"):
    openai.api_key = api_key
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

# Sidebar config
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

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

if webrtc_ctx.audio_processor and webrtc_ctx.state.playing:
    audio_processor = webrtc_ctx.audio_processor
    audio_data = b"".join(list(audio_processor.audio_queue.queue))

    if audio_data:
        prompt = convert_speech_to_text(audio_data, openai_api_key)
        st.write(f"You (transcribed): {prompt}")

        openai.api_key = openai_api_key
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",  # Use GPT-4-turbo
            messages=st.session_state.messages
        )
        msg = response.choices[0].message['content']
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.write(f"Assistant: {msg}")

        # Convert response to audio and play it
        audio_content = convert_text_to_speech(msg, openai_api_key)
        audio_str = audio_bytes_to_base64(audio_content)
        st.audio(audio_str, format="audio/mp3")

# Display chat history
for msg in st.session_state.messages:
    st.write(f"{msg['role'].capitalize()}: {msg['content']}")

# Error handling for the WebRTC context
if webrtc_ctx.state.iceConnectionState == "failed":
    st.error("WebRTC connection failed. Please check your network settings and try again.")
