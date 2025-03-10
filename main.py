import streamlit as st
import tempfile
import os
import time
from openai import OpenAI
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# Set page configuration
st.set_page_config(page_title="Voice Translator", page_icon="🎙", layout="wide")

# Initialize paths and variables
AUDIO_FILE_PATH = "recorded_audio.wav"
print("Application starting...")

# API Key management
st.sidebar.title("API Configuration")

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", value=st.session_state.api_key
)
if api_key:
    st.session_state.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    print("API key updated")

# App title and description
st.title("🎙 Voice Translator")
st.markdown(
    "Record your voice, convert it to text, and translate it to different languages."
)

# Check if API key is provided
if not st.session_state.api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to use this app.")
    st.stop()



def get_openai_client():
    if not st.session_state.api_key:
        return None
    print("Initializing OpenAI client")
    return OpenAI(api_key=st.session_state.api_key)


client = get_openai_client()
if not client:
    st.error("Failed to initialize OpenAI client")
    st.stop()

# Language selection
languages = {
    "Arabic": "ar",
    "Chinese": "zh",
    "Dutch": "nl",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Portuguese": "pt",
    "Russian": "ru",
    "Spanish": "es",
    "Swedish": "sv",
    "Turkish": "tr",
}


# Audio recording function
def record_audio(duration=5, sample_rate=16000):
    print(f"Starting audio recording for {duration} seconds")
    st.write("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    progress_bar = st.progress(0)

    for i in range(100):
        time.sleep(duration / 100)
        progress_bar.progress(i + 1)

    sd.wait()
    progress_bar.empty()
    st.success("Recording completed!")
    print("Recording completed successfully")

    return recording, sample_rate


# Save recording to a file
def save_audio(recording, sample_rate, file_path=AUDIO_FILE_PATH):
    print(f"Saving audio to {file_path}")
    # Remove existing file if it exists
    if os.path.exists(file_path):
        print(f"Removing existing audio file at {file_path}")
        os.remove(file_path)

    # Save the new recording
    write(file_path, sample_rate, recording)
    print(f"Audio saved successfully to {file_path}")
    return file_path


# Transcribe audio using OpenAI Whisper
def transcribe_audio(audio_file_path):
    print(f"Transcribing audio from {audio_file_path}")
    try:
        with open(audio_file_path, "rb") as audio_file:
            print("Opening audio file for transcription")
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
        print(f"Transcription successful: {transcription.text[:50]}...")
        return transcription.text
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        raise


# Translate text using OpenAI
def translate_text(text, target_language):
    print(f"Translating text to {target_language}")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a translator. Translate the following text to {target_language}. Only respond with the translated text, nothing else.",
                },
                {"role": "user", "content": text},
            ],
        )
        translation = response.choices[0].message.content
        print(f"Translation to {target_language} successful: {translation[:50]}...")
        return translation
    except Exception as e:
        print(f"Error in translation to {target_language}: {str(e)}")
        raise


# Main app layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Recording Options")

    duration = st.slider(
        "Recording Duration (seconds)", min_value=3, max_value=30, value=5
    )

    selected_languages = st.multiselect(
        "Select languages for translation",
        options=list(languages.keys()),
        default=["Spanish", "French", "German"],
    )

    record_button = st.button("Start Recording")

    if "transcription" in st.session_state:
        st.subheader("Original Text")
        st.write(st.session_state.transcription)

with col2:
    if "translations" in st.session_state and st.session_state.translations:
        st.subheader("Translations")
        for lang, translation in st.session_state.translations.items():
            with st.expander(f"{lang} Translation"):
                st.write(translation)

# Record audio when button is clicked
if record_button:
    if not selected_languages:
        st.error("Please select at least one language for translation.")
        print("Error: No languages selected")
    else:
        print("Starting recording and processing workflow")
        with st.spinner("Processing your voice..."):
            try:
                # Record audio
                print("Calling record_audio function")
                recording, sample_rate = record_audio(duration)

                # Save audio to file (overwriting any existing file)
                print("Calling save_audio function")
                audio_file_path = save_audio(recording, sample_rate)

                # Transcribe audio
                print("Calling transcribe_audio function")
                transcription = transcribe_audio(audio_file_path)
                st.session_state.transcription = transcription
                print(f"Full transcription: {transcription}")

                # Translate to selected languages
                print("Starting translations")
                translations = {}
                translation_progress = st.progress(0)

                for i, lang_name in enumerate(selected_languages):
                    print(
                        f"Translating to {lang_name} ({i+1}/{len(selected_languages)})"
                    )
                    translation = translate_text(
                        transcription, lang_name
                    )
                    translations[lang_name] = translation
                    translation_progress.progress((i + 1) / len(selected_languages))

                st.session_state.translations = translations
                translation_progress.empty()
                print("All translations completed successfully")

            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                print(f"ERROR: {error_msg}")

# Add some usage instructions
with st.expander("How to use this app"):
    st.markdown(
        """
    1. Enter your OpenAI API key in the sidebar
    2. Select the languages you want to translate to
    3. Set the recording duration
    4. Click 'Start Recording' and speak
    5. Wait for the processing to complete
    6. View your transcription and translations
    """
    )

# Display audio file information if it exists
if os.path.exists(AUDIO_FILE_PATH):
    file_stats = os.stat(AUDIO_FILE_PATH)
    st.sidebar.write(f"Audio file size: {file_stats.st_size / 1024:.2f} KB")
    st.sidebar.write(f"Last modified: {time.ctime(file_stats.st_mtime)}")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and OpenAI API")

print("Application fully loaded and ready")


# import streamlit as st
# import tempfile
# import os
# import time
# from transformers import pipeline
# import numpy as np
# import sounddevice as sd
# from scipy.io.wavfile import write

# # Set page configuration
# st.set_page_config(page_title="Voice Translator", page_icon="🎙", layout="wide")

# # Initialize paths and variables
# AUDIO_FILE_PATH = "recorded_audio.wav"
# print("Application starting...")

# # API Key management
# st.sidebar.title("API Configuration")

# if "api_key" not in st.session_state:
#     st.session_state.api_key = ""

# api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.api_key)
# if api_key:
#     st.session_state.api_key = api_key
#     os.environ["OPENAI_API_KEY"] = api_key
#     print("API key updated")

# # App title and description
# st.title("🎙 Voice Translator")
# st.markdown("Record your voice, convert it to text, and translate it to different languages.")

# # Check if API key is provided
# if not st.session_state.api_key:
#     st.warning("Please enter your OpenAI API key in the sidebar to use this app.")
#     st.stop()

# # Initialize Whisper model using transformers
# @st.cache_resource
# def load_whisper_model():
#     print("Loading Whisper model")
#     return pipeline("automatic-speech-recognition", model="openai/whisper-base")

# model = load_whisper_model()

# # Language selection
# languages = {
#     "Arabic": "ar",
#     "Chinese": "zh",
#     "Dutch": "nl",
#     "English": "en",
#     "French": "fr",
#     "German": "de",
#     "Hindi": "hi",
#     "Italian": "it",
#     "Japanese": "ja",
#     "Korean": "ko",
#     "Portuguese": "pt",
#     "Russian": "ru",
#     "Spanish": "es",
#     "Swedish": "sv",
#     "Turkish": "tr"
# }

# # Audio recording function
# def record_audio(duration=5, sample_rate=16000):
#     print(f"Starting audio recording for {duration} seconds")
#     st.write("Recording...")
#     recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
#     progress_bar = st.progress(0)

#     for i in range(100):
#         time.sleep(duration / 100)
#         progress_bar.progress(i + 1)

#     sd.wait()
#     progress_bar.empty()
#     st.success("Recording completed!")
#     print("Recording completed successfully")

#     return recording, sample_rate

# # Save recording to a file
# def save_audio(recording, sample_rate, file_path=AUDIO_FILE_PATH):
#     print(f"Saving audio to {file_path}")
#     # Remove existing file if it exists
#     if os.path.exists(file_path):
#         print(f"Removing existing audio file at {file_path}")
#         os.remove(file_path)

#     # Save the new recording
#     write(file_path, sample_rate, recording)
#     print(f"Audio saved successfully to {file_path}")
#     return file_path

# # Transcribe audio using Whisper
# def transcribe_audio(audio_file_path):
#     print(f"Transcribing audio from {audio_file_path}")
#     try:
#         result = model(audio_file_path)
#         transcription = result["text"]
#         print(f"Transcription successful: {transcription[:50]}...")
#         return transcription
#     except Exception as e:
#         print(f"Error in transcription: {str(e)}")
#         raise

# # Translate text using OpenAI
# def translate_text(text, target_language):
#     print(f"Translating text to {target_language}")
#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": f"You are a translator. Translate the following text to {target_language}. Only respond with the translated text, nothing else."},
#                 {"role": "user", "content": text}
#             ]
#         )
#         translation = response.choices[0].message.content
#         print(f"Translation to {target_language} successful: {translation[:50]}...")
#         return translation
#     except Exception as e:
#         print(f"Error in translation to {target_language}: {str(e)}")
#         raise

# # Main app layout
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Recording Options")

#     duration = st.slider("Recording Duration (seconds)", min_value=3, max_value=30, value=5)

#     selected_languages = st.multiselect(
#         "Select languages for translation",
#         options=list(languages.keys()),
#         default=["Spanish", "French", "German"]
#     )

#     record_button = st.button("Start Recording")

#     if "transcription" in st.session_state:
#         st.subheader("Original Text")
#         st.write(st.session_state.transcription)

# with col2:
#     if "translations" in st.session_state and st.session_state.translations:
#         st.subheader("Translations")
#         for lang, translation in st.session_state.translations.items():
#             with st.expander(f"{lang} Translation"):
#                 st.write(translation)

# # Record audio when button is clicked
# if record_button:
#     if not selected_languages:
#         st.error("Please select at least one language for translation.")
#         print("Error: No languages selected")
#     else:
#         print("Starting recording and processing workflow")
#         with st.spinner("Processing your voice..."):
#             try:
#                 # Record audio
#                 print("Calling record_audio function")
#                 recording, sample_rate = record_audio(duration)

#                 # Save audio to file (overwriting any existing file)
#                 print("Calling save_audio function")
#                 audio_file_path = save_audio(recording, sample_rate)

#                 # Transcribe audio
#                 print("Calling transcribe_audio function")
#                 transcription = transcribe_audio(audio_file_path)
#                 st.session_state.transcription = transcription
#                 print(f"Full transcription: {transcription}")

#                 # Translate to selected languages
#                 print("Starting translations")
#                 translations = {}
#                 translation_progress = st.progress(0)

#                 for i, lang_name in enumerate(selected_languages):
#                     print(f"Translating to {lang_name} ({i+1}/{len(selected_languages)})")
#                     translation = translate_text(transcription, lang_name)
#                     translations[lang_name] = translation
#                     translation_progress.progress((i + 1) / len(selected_languages))

#                 st.session_state.translations = translations
#                 translation_progress.empty()
#                 print("All translations completed successfully")

#             except Exception as e:
#                 error_msg = f"An error occurred: {str(e)}"
#                 st.error(error_msg)
#                 print(f"ERROR: {error_msg}")

# # Add some usage instructions
# with st.expander("How to use this app"):
#     st.markdown("""
#     1. Enter your OpenAI API key in the sidebar
#     2. Select the languages you want to translate to
#     3. Set the recording duration
#     4. Click 'Start Recording' and speak
#     5. Wait for the processing to complete
#     6. View your transcription and translations
#     """)

# # Display audio file information if it exists
# if os.path.exists(AUDIO_FILE_PATH):
#     file_stats = os.stat(AUDIO_FILE_PATH)
#     st.sidebar.write(f"Audio file size: {file_stats.st_size / 1024:.2f} KB")
#     st.sidebar.write(f"Last modified: {time.ctime(file_stats.st_mtime)}")

# # Add footer
# st.markdown("---")
# st.markdown("Built with Streamlit, Transformers, and OpenAI GPT")

# print("Application fully loaded and ready")