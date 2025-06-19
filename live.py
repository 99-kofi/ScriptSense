import streamlit as st
from PIL import Image
import cv2
import numpy as np
import google.generativeai as genai
# import tempfile # No longer needed for image saving if passing PIL Image directly
import os

# --- CONFIG ---
st.set_page_config(page_title="ScriptSense AI", layout="wide")

# --- API KEY ---
# Option 1: Use Streamlit secrets (recommended for deployed apps)
# Create a .streamlit/secrets.toml file with:
# GEMINI_API_KEY = "your_actual_api_key"
try:
    GEMINI_API_KEY = st.secrets.get(["GEMINI_API_KEY1"])
except (FileNotFoundError, KeyError):
    # Option 2: Use environment variable (good for local development)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY1")
    if not GEMINI_API_KEY:
        # Option 3: Fallback to a text input if no other method works (less secure for shared use)
        st.warning("GEMINI_API_KEY not found in secrets or environment. Please enter it below.")
        GEMINI_API_KEY = st.text_input("Enter your Gemini API Key:", type="password")

if not GEMINI_API_KEY:
    st.error("Gemini API Key is not configured. Please set it up to use the app.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # --- INIT GEMINI MODEL ---
    # Check if models are already in session state to avoid re-initializing
    if "model_vision" not in st.session_state:
        st.session_state.model_vision = genai.GenerativeModel("gemini-2.0-flash")
    if "model_text" not in st.session_state:
        st.session_state.model_text = genai.GenerativeModel("gemini-2.0-flash")

    model_vision = st.session_state.model_vision
    model_text = st.session_state.model_text

except Exception as e:
    st.error(f"Error initializing Gemini Models or configuring API key: {e}")
    st.error("Please ensure your API key is valid and has access to 'gemini-pro-vision' and 'gemini-pro' models.")
    st.stop()


# --- SESSION STATE INITIALIZATION ---
if "captured_image_pil" not in st.session_state:
    st.session_state.captured_image_pil = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False
if "previous_language" not in st.session_state:
    st.session_state.previous_language = None


# --- FUNCTIONS ---
def process_image_with_gemini(pil_image, lang_opt):
    """Sends the image to Gemini Vision for analysis."""
    try:
        prompt = f"Extract all the handwritten text from this image. Then translate it to {lang_opt}. Also summarize and analyze it for meaning, intent, and any action points."
        # Gemini Pro Vision can accept PIL Image objects directly in the list
        response = model_vision.generate_content([prompt, pil_image])
        return response.text
    except Exception as e:
        st.error(f"Error during Gemini Vision processing: {e}")
        # Check for specific safety/blockage reasons
        try:
            if response.prompt_feedback.block_reason:
                st.error(f"Content blocked due to: {response.prompt_feedback.block_reason}")
        except: # If response object itself is the issue or doesn't have prompt_feedback
            pass
        return None

def get_followup_response(previous_text, user_question):
    """Gets a follow-up response from Gemini Text model."""
    try:
        prompt = f"Given the previous text analysis:\n\n\"\"\"\n{previous_text}\n\"\"\"\n\nAnswer this question: {user_question}"
        response = model_text.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error during Gemini Text follow-up: {e}")
        try:
            if response.prompt_feedback.block_reason:
                st.error(f"Content blocked due to: {response.prompt_feedback.block_reason}")
        except:
            pass
        return None

# --- UI ---
st.title("✍️ ScriptSense AI: Handwriting to Intelligence")
st.markdown("Capture or upload an image of handwritten text. ScriptSense will extract, translate, analyze it, and answer your follow-up questions.")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🖼️ Input Image")
    language_option = st.selectbox(
        "Target Language for Translation",
        ["English", "French", "Spanish", "German", "Twi", "Hausa", "Arabic", "Chinese", "Japanese", "Korean"],
        key="language_select"
    )

    # Option to upload image
    uploaded_file = st.file_uploader("Upload an image of handwriting", type=["png", "jpg", "jpeg"])

    # Option to use camera
    if st.button("📷 Activate Camera"):
        st.session_state.show_camera = not st.session_state.show_camera # Toggle
        if not st.session_state.show_camera: # if we just turned it off, clear captured image
            st.session_state.captured_image_pil = None
            st.session_state.analysis_result = None


    if st.session_state.show_camera:
        st.info("Camera is active. Position your handwritten note.")
        camera_img_file_buffer = st.camera_input("Take a picture")
        if camera_img_file_buffer is not None:
            try:
                pil_image_cam = Image.open(camera_img_file_buffer)
                st.session_state.captured_image_pil = pil_image_cam
                st.session_state.analysis_result = None # Clear previous analysis
                st.session_state.show_camera = False # Turn off camera after capture
                st.rerun() # Rerun to update UI immediately
            except Exception as e:
                st.error(f"Could not process camera image: {e}")
                st.session_state.captured_image_pil = None

    # Process uploaded file if available
    if uploaded_file is not None:
        try:
            pil_image_upload = Image.open(uploaded_file)
            st.session_state.captured_image_pil = pil_image_upload # Prioritize uploaded if both exist
            st.session_state.analysis_result = None # Clear previous analysis
            st.session_state.show_camera = False # Ensure camera is off
        except Exception as e:
            st.error(f"Could not process uploaded image: {e}")
            st.session_state.captured_image_pil = None


    if st.session_state.captured_image_pil:
        st.image(st.session_state.captured_image_pil, caption="Selected Image", use_container_width=True)
        if st.button("Clear Image & Analysis", key="clear_image"):
            st.session_state.captured_image_pil = None
            st.session_state.analysis_result = None
            st.session_state.show_camera = False
            st.rerun()

# --- Analysis and Display Column ---
with col2:
    if st.session_state.captured_image_pil:
        # Re-analyze if image exists and (no previous result OR language changed)
        language_changed = (st.session_state.previous_language != language_option)
        if st.session_state.analysis_result is None or language_changed:
            with st.spinner(f"Analyzing handwriting & translating to {language_option}..."):
                analysis = process_image_with_gemini(st.session_state.captured_image_pil, language_option)
                if analysis:
                    st.session_state.analysis_result = analysis
                    st.session_state.previous_language = language_option # Store current language
                else:
                    # Keep old result if new analysis failed, unless language changed
                    if language_changed:
                         st.session_state.analysis_result = "Analysis failed. Please try a different image or check logs."


        if st.session_state.analysis_result:
            st.subheader("📋 Gemini Interpretation")
            st.text_area("Result", value=st.session_state.analysis_result, height=300, key="analysis_output_area")

            st.download_button(
                "⬇️ Download Result",
                st.session_state.analysis_result,
                file_name=f"ScriptSense_Analysis_{language_option}.txt",
                mime="text/plain"
            )

            st.subheader("💬 Ask a follow-up question")
            user_q = st.text_input("Your question about the text:", key="follow_up_q")
            if user_q and st.button("Get Answer", key="get_followup_answer"):
                with st.spinner("Generating follow-up response..."):
                    followup_text = get_followup_response(st.session_state.analysis_result, user_q)
                    if followup_text:
                        st.success(followup_text)
                    else:
                        st.warning("Could not get a follow-up response.")
    else:
        st.info("Upload an image or use the camera to get started.")


# Footer
st.markdown("---")
st.markdown("Developed with ❤️ by WAIT Technologies | Powered by Gemini AI")
