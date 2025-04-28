import streamlit as st
import tempfile
import os
import time
from openai import OpenAI
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read
import base64
from gtts import gTTS
import langid
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import datetime

# Set page configuration with a custom theme
st.set_page_config(
    page_title="Advanced Voice Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize paths and variables
AUDIO_FILE_PATH = "recorded_audio.wav"
TEMP_DIR = tempfile.gettempdir()
HISTORY_FILE = os.path.join(TEMP_DIR, "translation_history.csv")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px #BBB;
    }
    .sub-header {
        font-size: 1.8rem !important;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #555;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #0D47A1;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .recording-button {
        background-color: #f44336 !important;
    }
    .recording-button:hover {
        background-color: #d32f2f !important;
    }
    .feature-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .translation-box {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #1E88E5, transparent);
        margin: 1rem 0;
    }
    /* Custom styles for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        height: auto;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    /* Recording animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    .recording-pulse {
        animation: pulse 1.5s infinite;
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "translations" not in st.session_state:
    st.session_state.translations = {}
if "translation_history" not in st.session_state:
    st.session_state.translation_history = []
if "detected_language" not in st.session_state:
    st.session_state.detected_language = ""
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "recording_state" not in st.session_state:
    st.session_state.recording_state = "idle"  # idle, recording, processing
if "audio_recorded" not in st.session_state:
    st.session_state.audio_recorded = False
if "audio_file_path" not in st.session_state:
    st.session_state.audio_file_path = None

# Sidebar for API key and settings
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/translate-app.png", width=100)
    st.title("Settings")
    
    api_key = st.text_input(
        "OpenAI API Key", type="password", value=st.session_state.api_key
    )
    if api_key:
        st.session_state.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.divider()
    
    st.subheader("⚙️ App Settings")
    
    # Theme selection
    theme = st.selectbox(
        "Select Theme",
        ["Blue (Default)", "Dark", "Light", "Purple", "Green"],
    )
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        whisper_model = st.selectbox(
            "Whisper Model",
            ["whisper-1"],
            index=0,
            help="Select the OpenAI Whisper model for transcription"
        )
        
        gpt_model = st.selectbox(
            "GPT Model for Translation",
            ["gpt-3.5-turbo", "gpt-4"],
            index=0,
            help="Select the GPT model for translation"
        )
        
        show_confidence = st.checkbox("Show Confidence Scores", value=False)
        save_history = st.checkbox("Save Translation History", value=True)

    st.divider()

    # Audio controls
    st.subheader("🔊 Audio Settings")
    auto_play = st.checkbox("Auto-play Translations", value=False)
    speech_rate = st.slider("Speech Rate", min_value=0.5, max_value=1.5, value=1.0, step=0.1)

# App header
st.markdown('<h1 class="main-header">🌐 Advanced Voice Translator</h1>', unsafe_allow_html=True)
st.markdown('<p class="info-text" style="text-align: center;">Record your voice, get instant transcription, and translate to multiple languages with text-to-speech capabilities</p>', unsafe_allow_html=True)

# Check if API key is provided
if not st.session_state.api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to use all features of this app.")
    # Still show the UI, but disable functionality that requires API key
    api_available = False
else:
    api_available = True

def get_openai_client():
    if not st.session_state.api_key:
        return None
    return OpenAI(api_key=st.session_state.api_key)

client = get_openai_client() if api_available else None

# Language selection with flags
languages = {
    "Arabic": {"code": "ar", "flag": "🇸🇦"},
    "Chinese": {"code": "zh", "flag": "🇨🇳"},
    "Dutch": {"code": "nl", "flag": "🇳🇱"},
    "English": {"code": "en", "flag": "🇬🇧"},
    "French": {"code": "fr", "flag": "🇫🇷"},
    "German": {"code": "de", "flag": "🇩🇪"},
    "Hindi": {"code": "hi", "flag": "🇮🇳"},
    "Italian": {"code": "it", "flag": "🇮🇹"},
    "Japanese": {"code": "ja", "flag": "🇯🇵"},
    "Korean": {"code": "ko", "flag": "🇰🇷"},
    "Portuguese": {"code": "pt", "flag": "🇵🇹"},
    "Russian": {"code": "ru", "flag": "🇷🇺"},
    "Spanish": {"code": "es", "flag": "🇪🇸"},
    "Swedish": {"code": "sv", "flag": "🇸🇪"},
    "Turkish": {"code": "tr", "flag": "🇹🇷"},
}

# Create language options with flags
language_options = [f"{lang_data['flag']} {lang_name}" for lang_name, lang_data in languages.items()]

# Helper functions

def auto_detect_language(text):
    """Detect the language of the input text"""
    lang, confidence = langid.classify(text)
    # Map the language code to our language names
    for name, data in languages.items():
        if data["code"] == lang:
            return name, confidence
    return "Unknown", confidence

def create_audio_player(audio_bytes, autoplay=False):
    """Create an HTML audio player with the provided audio bytes"""
    b64 = base64.b64encode(audio_bytes).decode()
    auto_attr = "autoplay" if autoplay else ""
    return f'<audio {auto_attr} controls><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'

def text_to_speech(text, lang_code, rate=1.0):
    """Convert text to speech using gTTS"""
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False if rate >= 1.0 else True)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.read()
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

def save_to_history(source_text, source_lang, translations, timestamp=None):
    """Save the translation to history"""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_entry = {
        "timestamp": timestamp,
        "source_text": source_text,
        "source_language": source_lang,
    }
    
    for lang, text in translations.items():
        new_entry[f"translation_{lang}"] = text
    
    st.session_state.translation_history.append(new_entry)
    
    # Save to CSV
    if save_history:
        try:
            df = pd.DataFrame(st.session_state.translation_history)
            df.to_csv(HISTORY_FILE, index=False)
        except Exception as e:
            st.warning(f"Could not save history: {str(e)}")

def record_audio(duration=5, sample_rate=16000):
    """Record audio from the microphone with proper error handling"""
    try:
        # Show recording status
        status_placeholder = st.empty()
        status_placeholder.markdown('<p class="recording-pulse">● Recording in progress...</p>', unsafe_allow_html=True)
        
        # Create a progress bar
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        
        # Start recording
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        
        # Update progress while recording
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(duration / 100)
        
        # Wait for recording to complete
        sd.wait()
        
        # Clean up UI elements
        progress_placeholder.empty()
        status_placeholder.empty()
        
        # Check if recording contains data
        if np.max(np.abs(recording)) < 0.01:
            st.warning("Recording volume is very low. Please speak louder or check your microphone.")
        
        # Display a waveform of the recorded audio
        fig = go.Figure(go.Scatter(
            y=recording.flatten()[:min(1000, len(recording.flatten()))],
            mode='lines',
            line=dict(color='#1E88E5', width=1),
            fill='tozeroy',
            fillcolor='rgba(30, 136, 229, 0.3)'
        ))
        fig.update_layout(
            height=100,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return recording, sample_rate
    
    except Exception as e:
        st.error(f"Error while recording: {str(e)}")
        return None, None

def save_audio(recording, sample_rate, file_path=AUDIO_FILE_PATH):
    """Save the recorded audio to a file with proper error handling"""
    try:
        # Remove existing file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Save the new recording
        write(file_path, sample_rate, recording)
        
        # Verify file was created successfully
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return file_path
        else:
            st.error("Failed to save audio file.")
            return None
    
    except Exception as e:
        st.error(f"Error saving audio: {str(e)}")
        return None

def transcribe_audio(audio_file_path, model="whisper-1"):
    """Transcribe audio using OpenAI Whisper"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=model, file=audio_file
            )
        return transcription.text
    except Exception as e:
        raise Exception(f"Transcription error: {str(e)}")

def translate_text(text, target_language, model="gpt-3.5-turbo"):
    """Translate text using OpenAI GPT"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a translator. Translate the following text to {target_language}. Only respond with the translated text, nothing else.",
                },
                {"role": "user", "content": text},
            ],
        )
        translation = response.choices[0].message.content
        return translation
    except Exception as e:
        raise Exception(f"Translation error: {str(e)}")

def upload_audio_file():
    """Process an uploaded audio file"""
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            upload_path = tmp.name
        
        # Play the audio file
        st.audio(uploaded_file, format=f"audio/{os.path.splitext(uploaded_file.name)[1][1:]}")
        st.success("Audio file uploaded successfully!")
        return upload_path
    
    return None

# Main application tabs
tabs = st.tabs(["🎙️ Voice Translator", "💬 Conversation Mode", "📂 History", "📊 Analytics"])

# 1. Voice Translator Tab
with tabs[0]:
    st.markdown('<h2 class="sub-header">Record or Upload Audio</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        
        # Audio input options
        input_method = st.radio(
            "Choose input method",
            ["Record from microphone", "Upload audio file"],
            horizontal=True
        )
        
        # Reset audio file path when input method changes
        if input_method != st.session_state.get("last_input_method", None):
            st.session_state.audio_file_path = None
            st.session_state.audio_recorded = False
            st.session_state.last_input_method = input_method
        
        audio_file_path = None
        
        if input_method == "Record from microphone":
            duration = st.slider(
                "Recording Duration (seconds)", min_value=3, max_value=30, value=5
            )
            
            col_rec1, col_rec2 = st.columns([1, 1])
            with col_rec1:
                record_button = st.button("🎙️ Start Recording", key="record_btn")
            with col_rec2:
                auto_detect = st.checkbox("Auto-detect language", value=True)
            
            # Handle recording
            if record_button and api_available:
                st.session_state.recording_state = "recording"
                recording, sample_rate = record_audio(duration)
                
                if recording is not None and sample_rate is not None:
                    # Save the recording
                    audio_file_path = save_audio(recording, sample_rate)
                    
                    if audio_file_path:
                        st.session_state.audio_recorded = True
                        st.session_state.audio_file_path = audio_file_path
                        
                        # Let the user listen to the recording
                        st.audio(audio_file_path, format="audio/wav")
                        st.success("Recording completed and saved!")
                    else:
                        st.error("Failed to save the audio recording.")
                
                st.session_state.recording_state = "idle"
            
        else:  # Upload audio file
            uploaded_path = upload_audio_file()
            if uploaded_path:
                audio_file_path = uploaded_path
                st.session_state.audio_recorded = True
                st.session_state.audio_file_path = audio_file_path
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Translation options
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.subheader("Translation Options")
        
        # Create a more visual language selector with flags
        selected_languages_with_flags = st.multiselect(
            "Select languages for translation",
            options=language_options,
            default=[
                f"{languages['Spanish']['flag']} Spanish", 
                f"{languages['French']['flag']} French"
            ]
        )
        
        # Extract language names from selections with flags
        selected_languages = [lang.split(" ", 1)[1] for lang in selected_languages_with_flags]
        
        # Disable translate button if no audio is recorded
        translate_disabled = not (st.session_state.audio_recorded and 
                                 st.session_state.audio_file_path and 
                                 len(selected_languages) > 0)
        
        translate_button = st.button("🔄 Translate", key="translate_btn", 
                                    disabled=translate_disabled or not api_available)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        if st.session_state.transcription:
            st.markdown('<div class="translation-box">', unsafe_allow_html=True)
            st.subheader("Original Text")
            
            if st.session_state.detected_language:
                st.markdown(f"*Detected language: {st.session_state.detected_language}*")
            
            st.markdown(f"<p style='font-size: 1.2rem;'>{st.session_state.transcription}</p>", unsafe_allow_html=True)
            
            # Add text-to-speech for the original text
            if st.session_state.detected_language in languages:
                lang_code = languages[st.session_state.detected_language]["code"]
                tts_button = st.button("🔊 Listen", key="listen_original")
                if tts_button:
                    audio_bytes = text_to_speech(st.session_state.transcription, lang_code, speech_rate)
                    if audio_bytes:
                        st.markdown(create_audio_player(audio_bytes, auto_play), unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Process audio when translate button is clicked
    if translate_button and st.session_state.audio_file_path and api_available:
        if not selected_languages:
            st.error("Please select at least one language for translation.")
        else:
            st.session_state.recording_state = "processing"
            with st.spinner("Processing audio..."):
                try:
                    # Transcribe audio
                    transcription = transcribe_audio(st.session_state.audio_file_path, model=whisper_model)
                    if transcription:
                        st.session_state.transcription = transcription
                        
                        # Detect language if auto-detect is enabled
                        if auto_detect:
                            detected_lang, confidence = auto_detect_language(transcription)
                            st.session_state.detected_language = detected_lang
                            source_lang = detected_lang
                        else:
                            source_lang = "Unknown"
                        
                        # Translate to selected languages
                        translations = {}
                        translation_progress = st.progress(0)
                        
                        for i, lang_name in enumerate(selected_languages):
                            translation = translate_text(transcription, lang_name, model=gpt_model)
                            if translation:
                                translations[lang_name] = translation
                            translation_progress.progress((i + 1) / len(selected_languages))
                        
                        st.session_state.translations = translations
                        translation_progress.empty()
                        
                        # Save to history
                        if save_history:
                            save_to_history(transcription, source_lang, translations)
                        
                        # Add to conversation history
                        st.session_state.conversation_history.append({
                            "type": "original",
                            "text": transcription,
                            "language": source_lang
                        })
                        for lang, text in translations.items():
                            st.session_state.conversation_history.append({
                                "type": "translation",
                                "text": text,
                                "language": lang
                            })
                        
                        st.rerun()  # Changed from st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                
                st.session_state.recording_state = "idle"
    
    # Display translations
    if st.session_state.translations:
        st.markdown('<h2 class="sub-header">Translations</h2>', unsafe_allow_html=True)
        
        for lang, translation in st.session_state.translations.items():
            st.markdown(f'<div class="translation-box">', unsafe_allow_html=True)
            st.subheader(f"{languages.get(lang, {}).get('flag', '')} {lang}")
            st.markdown(f"<p style='font-size: 1.2rem;'>{translation}</p>", unsafe_allow_html=True)
            
            # Add text-to-speech for each translation
            if lang in languages:
                lang_code = languages[lang]["code"]
                col1, col2 = st.columns([1, 4])
                with col1:
                    tts_button = st.button(f"🔊 Listen", key=f"listen_{lang}")
                with col2:
                    copy_button = st.button(f"📋 Copy", key=f"copy_{lang}")
                
                if tts_button:
                    audio_bytes = text_to_speech(translation, lang_code, speech_rate)
                    if audio_bytes:
                        st.markdown(create_audio_player(audio_bytes, auto_play), unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Add example phrases for quick testing (moved outside the translations section)
    with st.expander("Quick Test Phrases"):
        st.markdown("Click on any phrase to use it as your source text:")

        example_phrases = [
            "Hello, how are you today?",
            "I would like to order some food.",
            "Where is the nearest train station?",
            "Can you help me find my hotel?",
            "What time does the museum open?"
        ]

        cols = st.columns(len(example_phrases))
        for i, phrase in enumerate(example_phrases):
            if cols[i].button(phrase, key=f"phrase_{i}"):
                st.session_state.transcription = phrase
                st.session_state.detected_language = "English"
                st.rerun()  # Changed from st.experimental_rerun()

# 2. Conversation Mode Tab
with tabs[1]:
    st.markdown('<h2 class="sub-header">Two-Way Conversation Mode</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.subheader("Conversation Settings")
        
        primary_lang = st.selectbox(
            "Your language",
            options=language_options,
            index=language_options.index(f"{languages['English']['flag']} English") if f"{languages['English']['flag']} English" in language_options else 0,
            key="primary_lang"
        )
        
        secondary_lang = st.selectbox(
            "Partner language",
            options=language_options,
            index=language_options.index(f"{languages['Spanish']['flag']} Spanish") if f"{languages['Spanish']['flag']} Spanish" in language_options else 0,
            key="secondary_lang"
        )
        
        # Extract language names
        primary_lang_name = primary_lang.split(" ", 1)[1]
        secondary_lang_name = secondary_lang.split(" ", 1)[1]
        
        conv_duration = st.slider(
            "Recording Duration (seconds)", 
            min_value=3, 
            max_value=30, 
            value=5,
            key="conv_duration"
        )
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            speak_primary = st.button(f"🎙️ Speak {primary_lang_name}", key="speak_primary")
        with col_b2:
            speak_secondary = st.button(f"🎙️ Speak {secondary_lang_name}", key="speak_secondary")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.subheader("Conversation History")
        
        conversation_container = st.container()
        with conversation_container:
            if not st.session_state.conversation_history:
                st.info("Your conversation will appear here.")
            else:
                for entry in st.session_state.conversation_history[-10:]:  # Show last 10 entries
                    if entry["type"] == "original":
                        st.markdown(f"<div style='background-color: #e3f2fd; border-radius: 15px; padding: 10px; margin: 5px 0;'>"
                                   f"<strong>{entry['language']}:</strong> {entry['text']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='background-color: #f1f8e9; border-radius: 15px; padding: 10px; margin: 5px 0;'>"
                                   f"<strong>{entry['language']}:</strong> {entry['text']}</div>", unsafe_allow_html=True)
        
        clear_conv = st.button("🗑️ Clear Conversation", key="clear_conv")
        if clear_conv:
            st.session_state.conversation_history = []
            st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process conversation mode recordings
    if (speak_primary or speak_secondary) and api_available:
        with st.spinner("Recording and translating..."):
            try:
                # Set recording state
                st.session_state.recording_state = "recording"
                
                # Record audio
                recording, sample_rate = record_audio(conv_duration)
                
                if recording is not None and sample_rate is not None:
                    # Save the recording
                    audio_file_path = save_audio(recording, sample_rate)
                    
                    if audio_file_path:
                        # Play back the recording
                        st.audio(audio_file_path, format="audio/wav")
                        
                        # Transcribe audio
                        transcription = transcribe_audio(audio_file_path, model=whisper_model)
                        
                        if transcription:
                            # Determine source and target languages
                            if speak_primary:
                                source_lang = primary_lang_name
                                target_lang = secondary_lang_name
                            else:
                                source_lang = secondary_lang_name
                                target_lang = primary_lang_name
                            
                            # Translate
                            translation = translate_text(transcription, target_lang, model=gpt_model)
                            
                            # Add to conversation history
                            st.session_state.conversation_history.append({
                                "type": "original",
                                "text": transcription,
                                "language": source_lang
                            })
                            
                            if translation:
                                st.session_state.conversation_history.append({
                                    "type": "translation",
                                    "text": translation,
                                    "language": target_lang
                                })
                                
                                # Auto play translation if enabled
                                if auto_play and target_lang in languages:
                                    target_lang_code = languages[target_lang]["code"]
                                    audio_bytes = text_to_speech(translation, target_lang_code, speech_rate)
                                    if audio_bytes:
                                        st.session_state["last_audio"] = audio_bytes
                                
                                # Save to history
                                if save_history:
                                    save_to_history(transcription, source_lang, {target_lang: translation})
                    else:
                        st.error("Failed to save the audio recording.")
                
                st.session_state.recording_state = "idle"
                st.rerun()  # Changed from st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Conversation error: {str(e)}")
                st.session_state.recording_state = "idle"
    
    # Play last audio if available
    if "last_audio" in st.session_state and auto_play:
        st.markdown(create_audio_player(st.session_state["last_audio"], True), unsafe_allow_html=True)
        del st.session_state["last_audio"]

# 3. History Tab
with tabs[2]:
    st.markdown('<h2 class="sub-header">Translation History</h2>', unsafe_allow_html=True)
    
    # Load history from file if exists
    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        try:
            history_df = pd.read_csv(HISTORY_FILE)
            if not st.session_state.translation_history:
                history_records = history_df.to_dict('records')
                st.session_state.translation_history = history_records
        except Exception as e:
            st.warning(f"Could not load history: {str(e)}")
    
    if not st.session_state.translation_history:
        st.info("No translation history available yet.")
    else:
        # Search and filter
        search_term = st.text_input("🔍 Search in history", "")
        
        # Filter by date range
        col1, col2 = st.columns(2)
        with col1:
            if 'timestamp' in st.session_state.translation_history[0]:
                dates = [datetime.datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S").date() 
                        for entry in st.session_state.translation_history]
                min_date = min(dates)
                max_date = max(dates)
                start_date = st.date_input("Start date", min_date)
            else:
                start_date = st.date_input("Start date")
                
        with col2:
            if 'timestamp' in st.session_state.translation_history[0]:
                end_date = st.date_input("End date", max_date)
            else:
                end_date = st.date_input("End date")
        
        # Filter history
        filtered_history = []
        for entry in st.session_state.translation_history:
            # Skip if no timestamp
            if 'timestamp' not in entry:
                continue
                
            # Check date range
            entry_date = datetime.datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S").date()
            if entry_date < start_date or entry_date > end_date:
                continue
                
            # Check search term
            if search_term and search_term.lower() not in entry['source_text'].lower():
                continue
                
            filtered_history.append(entry)
        
        if not filtered_history:
            st.info("No matching records found.")
        else:
            # Display history entries
            for idx, entry in enumerate(filtered_history):
                with st.expander(f"Entry {idx+1}: {entry['timestamp']}"):
                    st.markdown(f"**Original ({entry['source_language']}):** {entry['source_text']}")
                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                    
                    for key, value in entry.items():
                        if key.startswith('translation_'):
                            lang = key.replace('translation_', '')
                            st.markdown(f"**{lang}:** {value}")
                    
                    # Add replay and export options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Translate Again", key=f"retranslate_{idx}"):
                            st.session_state.transcription = entry['source_text']
                            st.session_state.detected_language = entry['source_language']
                            st.rerun()  # Changed from st.experimental_rerun()
                    with col2:
                        if st.button("📋 Copy All", key=f"copy_all_{idx}"):
                            pass  # Would use JavaScript in real app

# 4. Analytics Tab
with tabs[3]:
    st.markdown('<h2 class="sub-header">Usage Analytics</h2>', unsafe_allow_html=True)
    
    if not st.session_state.translation_history:
        st.info("No data available for analytics. Use the translator to generate data.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            # Language distribution chart
            st.subheader("Most Translated Languages")
            
            lang_counts = {}
            for entry in st.session_state.translation_history:
                for key in entry.keys():
                    if key.startswith('translation_'):
                        lang = key.replace('translation_', '')
                        lang_counts[lang] = lang_counts.get(lang, 0) + 1
            
            if lang_counts:
                lang_df = pd.DataFrame({
                    'Language': list(lang_counts.keys()),
                    'Count': list(lang_counts.values())
                })
                
                fig, ax = plt.subplots()
                bars = ax.bar(lang_df['Language'], lang_df['Count'], color='#1E88E5')
                ax.set_ylabel('Number of Translations')
                ax.set_xlabel('Language')
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.subheader("Usage Over Time")
            
            # Create time-series data
            if st.session_state.translation_history and 'timestamp' in st.session_state.translation_history[0]:
                dates = [datetime.datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S").date() 
                        for entry in st.session_state.translation_history]
                
                date_counts = {}
                for date in dates:
                    date_str = date.strftime("%Y-%m-%d")
                    date_counts[date_str] = date_counts.get(date_str, 0) + 1
                
                date_df = pd.DataFrame({
                    'Date': list(date_counts.keys()),
                    'Count': list(date_counts.values())
                })
                date_df['Date'] = pd.to_datetime(date_df['Date'])
                date_df = date_df.sort_values('Date')
                
                fig, ax = plt.subplots()
                ax.plot(date_df['Date'], date_df['Count'], marker='o', linestyle='-', color='#1E88E5')
                ax.set_ylabel('Number of Translations')
                ax.set_xlabel('Date')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No timestamp data available for time-series analysis.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Text length analysis
    st.markdown('<div class="feature-box">', unsafe_allow_html=True)
    st.subheader("Text Length Analysis")
    
    if st.session_state.translation_history:
        # Calculate text lengths
        source_lengths = [len(entry['source_text']) for entry in st.session_state.translation_history]
        
        # Calculate statistics
        avg_length = sum(source_lengths) / len(source_lengths)
        max_length = max(source_lengths)
        min_length = min(source_lengths)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Length", f"{avg_length:.1f} chars")
        col2.metric("Maximum Length", f"{max_length} chars")
        col3.metric("Minimum Length", f"{min_length} chars")
        
        # Create a histogram of text lengths
        fig, ax = plt.subplots()
        ax.hist(source_lengths, bins=10, color='#1E88E5', alpha=0.7)
        ax.set_xlabel('Text Length (characters)')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No data available for text length analysis.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
# Footer with additional information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### About")
    st.markdown("Advanced Voice Translator leverages OpenAI's Whisper and GPT models to provide high-quality speech recognition and translation.")

with col2:
    st.markdown("### Features")
    st.markdown("""
    - Speech-to-text transcription
    - Translation to multiple languages
    - Text-to-speech for translations
    - Conversation mode
    - History tracking and analytics
    """)

with col3:
    st.markdown("### Help")
    with st.expander("How to use this app"):
        st.markdown("""
        1. Enter your OpenAI API key in the sidebar
        2. Select languages for translation
        3. Record your voice or upload an audio file
        4. Get instant transcription and translations
        5. Use text-to-speech to hear the translations
        """)
    
    with st.expander("Troubleshooting"):
        st.markdown("""
        - Make sure your microphone is working properly
        - Check your API key if you encounter authentication errors
        - For large audio files, increase the processing timeout
        - Clear browser cache if the app is not loading correctly
        - If recording fails, try uploading an audio file instead
        """)