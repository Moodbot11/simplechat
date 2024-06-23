import os
import base64
import streamlit as st
import openai

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
def convert_speech_to_text(audio_file_path, api_key):
    openai.api_key = api_key
    with open(audio_file_path, "rb") as audio_file:
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

# Radio button to choose input method
input_method = st.radio("Choose input method:", ("Text Input", "Voice Input"))

# Handle text input
if input_method == "Text Input":
    prompt = st.text_input("You:")
    if prompt and openai_api_key:
        openai.api_key = openai_api_key
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.write(f"You: {prompt}")

        response = openai.ChatCompletion.create(
            model="gpt-4", messages=st.session_state.messages
        )
        msg = response.choices[0].message['content']
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.write(f"Assistant: {msg}")

        # Convert response to audio and play it
        audio_content = convert_text_to_speech(msg, openai_api_key)
        audio_str = audio_bytes_to_base64(audio_content)
        st.audio(audio_str, format="audio/mp3")

# Handle voice input
elif input_method == "Voice Input":
    uploaded_file = st.file_uploader("Upload your voice input (wav format)", type=["wav"])

    if uploaded_file is not None and openai_api_key:
        with open(uploaded_file.name, "wb") as file:
            file.write(uploaded_file.getbuffer())

        prompt = convert_speech_to_text(uploaded_file.name, openai_api_key)
        st.write(f"You (transcribed): {prompt}")

        openai.api_key = openai_api_key
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model="gpt-4o", messages=st.session_state.messages
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
