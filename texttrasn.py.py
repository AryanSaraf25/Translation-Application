import streamlit as st
import os
import time
from google import genai

# Set page configuration
st.set_page_config(page_title="Text Translator", page_icon="🌎", layout="wide")
print("Application starting...")

# API Key management
st.sidebar.title("API Configuration")
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

api_key = st.sidebar.text_input(
    "Google Gemini API Key", type="password", value=st.session_state.api_key
)

if api_key:
    st.session_state.api_key = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    print("API key updated")

# App title and description
st.title("🌎 Text Translator")
st.markdown("Enter text and translate it to different languages using Google Gemini.")

# Check if API key is provided
if not st.session_state.api_key:
    st.warning(
        "Please enter your Google Gemini API key in the sidebar to use this app."
    )
    st.stop()


# Initialize Gemini
def setup_gemini():
    if not st.session_state.api_key:
        return False
    try:
        print("Initializing Google Gemini")
        client = genai.Client(api_key=st.session_state.api_key)
        return client
    except Exception as e:
        print(f"Error initializing Gemini: {str(e)}")
        return False


client = setup_gemini()

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


# Translate text using Gemini
def translate_text(text, target_language):
    print(f"Translating text to {target_language}")
    try:
        prompt = f"Translate the following text to {target_language}. Only respond with the translated text, nothing else.\n\nText: {text}"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        translation = response.text.strip()
        print(f"Translation to {target_language} successful: {translation[:50]}...")
        return translation
    except Exception as e:
        print(f"Error in translation to {target_language}: {str(e)}")
        raise


# Initialize session state variables if they don't exist
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "selected_languages" not in st.session_state:
    st.session_state.selected_languages = ["Spanish", "French", "German"]
if "translations" not in st.session_state:
    st.session_state.translations = {}
if "is_translating" not in st.session_state:
    st.session_state.is_translating = False
if "should_translate" not in st.session_state:
    st.session_state.should_translate = False

# Input section (top)
st.subheader("Input Text")
input_text = st.text_area(
    "Enter text to translate",
    value=st.session_state.input_text,
    height=200,
    key="input_text_area",
)
st.session_state.input_text = input_text

# Language selection
selected_languages = st.multiselect(
    "Select languages for translation",
    options=list(languages.keys()),
    default=st.session_state.selected_languages,
    key="language_selector",
)
st.session_state.selected_languages = selected_languages

# Auto-translate toggle and manual translate button in the same row
col1, col2 = st.columns([3, 1])
with col1:
    auto_translate = st.checkbox(
        "Auto-translate",
        value=True,
        help="Automatically translate when text or languages change",
    )
with col2:
    manual_translate = st.button("Translate Now", use_container_width=True)

# Set flag to translate based on button or auto-translate
if manual_translate:
    st.session_state.should_translate = True
elif auto_translate and (
    input_text != st.session_state.get("last_translated_text", "")
    or set(selected_languages)
    != set(st.session_state.get("last_translated_languages", []))
):
    st.session_state.should_translate = True

# Translation logic
if st.session_state.should_translate and (
    (auto_translate and input_text and selected_languages) or manual_translate
):
    if not input_text:
        st.error("Please enter some text to translate.")
        print("Error: No input text provided")
    elif not selected_languages:
        st.error("Please select at least one language for translation.")
        print("Error: No languages selected")
    else:
        print("Starting translation workflow")

        # Reset the flag
        st.session_state.should_translate = False
        st.session_state.is_translating = True

        # Store what we're translating to avoid duplicate work
        st.session_state.last_translated_text = input_text
        st.session_state.last_translated_languages = selected_languages.copy()

        # Progress indicator
        progress_container = st.container()
        with progress_container:
            progress_text = st.empty()
            progress_bar = st.progress(0)

        try:
            # Clear previous translations if language selection has changed
            if set(st.session_state.translations.keys()) != set(selected_languages):
                st.session_state.translations = {}

            # Translate to selected languages
            print("Starting translations")
            for i, lang_name in enumerate(selected_languages):
                progress_text.text(
                    f"Translating to {lang_name}... ({i+1}/{len(selected_languages)})"
                )
                print(f"Translating to {lang_name} ({i+1}/{len(selected_languages)})")

                # Skip if already translated
                if (
                    lang_name in st.session_state.translations
                    and st.session_state.translations[lang_name]
                ):
                    continue

                # Translate
                translation = translate_text(input_text, lang_name)
                st.session_state.translations[lang_name] = translation

                # Update progress
                progress_bar.progress((i + 1) / len(selected_languages))

            # Clear progress indicators when done
            progress_text.empty()
            progress_bar.empty()
            print("All translations completed successfully")

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.error(error_msg)
            print(f"ERROR: {error_msg}")

        finally:
            st.session_state.is_translating = False

# Display translations section (only if there are translations)
if st.session_state.translations:
    st.markdown("---")
    st.subheader("Translations")

    # Create tabs for each translation
    if len(st.session_state.translations) > 0:
        tabs = st.tabs(list(st.session_state.translations.keys()))
        for i, (lang, translation) in enumerate(st.session_state.translations.items()):
            with tabs[i]:
                st.write(translation)
                copy_button = st.button(f"Copy {lang} translation", key=f"copy_{lang}")
                if copy_button:
                    st.toast(f"{lang} translation copied to clipboard!")
                    # Note: Actual clipboard functionality requires JavaScript;
                    # This is just a placeholder notification

# Add some usage instructions
with st.expander("How to use this app"):
    st.markdown(
        """
        1. Enter your Google Gemini API key in the sidebar
        2. Type or paste the text you want to translate
        3. Select the languages you want to translate to
        4. Enable auto-translate for automatic updates or click "Translate Now"
        5. View your translations in the tabs below after processing
        """
    )

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and Google Gemini API")
print("Application fully loaded and ready")
