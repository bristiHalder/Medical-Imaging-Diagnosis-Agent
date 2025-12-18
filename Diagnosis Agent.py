import os
import numpy as np
import streamlit as st
from io import BytesIO
from PIL import Image as PILImage
import cv2
import pydicom

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
from agno.run.agent import RunOutput


# =========================
# CONFIGURATION
# =========================
st.set_page_config(
    page_title="Medical Imaging Diagnosis Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = None


# =========================
# UTILITY FUNCTIONS
# =========================
def load_medical_image(uploaded_file):
    """Load JPG/PNG/DICOM safely"""
    if uploaded_file.name.lower().endswith(".dcm"):
        dicom = pydicom.dcmread(uploaded_file)
        image = dicom.pixel_array.astype(float)
        image = (image - image.min()) / (image.max() - image.min()) * 255
        return PILImage.fromarray(image.astype(np.uint8))
    return PILImage.open(uploaded_file)


def resize_image(image, target_width=500):
    w, h = image.size
    aspect_ratio = w / h
    new_height = int(target_width / aspect_ratio)
    return image.resize((target_width, new_height))


def is_blurry(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < 100


def create_agno_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return AgnoImage(content=buffer.getvalue())


@st.cache_resource
def get_medical_agent(api_key):
    return Agent(
        model=Gemini(
            id="gemini-2.5-pro",
            api_key=api_key
        ),
        tools=[DuckDuckGoTools()],
        markdown=True
    )


# =========================
# MEDICAL PROMPT
# =========================
MEDICAL_ANALYSIS_PROMPT = """
You are an AI-assisted medical imaging analysis system designed to support radiological review.

IMPORTANT SAFETY RULES:
- Do NOT provide definitive diagnoses.
- Use uncertainty-aware language: "suggestive of", "may indicate", "cannot rule out".
- This is NOT a certified medical device.

Structure your response as follows:

### 1. Image Type & Region
- Imaging modality (X-ray / MRI / CT / Ultrasound / Unknown)
- Anatomical region and orientation
- Image quality and limitations

### 2. Key Observations
- Systematic visual findings
- Any abnormal patterns or structures
- Location, size, symmetry, intensity
- Severity estimate: Normal / Mild / Moderate / Severe

### 3. AI-Assisted Radiological Impression
- Most likely interpretation (with confidence level)
- Differential considerations
- Any findings that may require urgent attention

### 4. Patient-Friendly Explanation
- Simple, non-technical explanation
- Reassuring and clear tone
- Clarify uncertainty and next steps

### 5. Research Context
Use web search to:
- Reference similar cases
- Mention general clinical management approaches
- Provide 2–3 reputable medical references

### 6. AI Limitations & Confidence
- Dependence on image quality
- No access to patient history
- Should be reviewed by a medical professional

Format using clear markdown headings and bullet points.
"""


# =========================
# SIDEBAR – API CONFIG
# =========================
with st.sidebar:
    st.title("Configuration")

    if not st.session_state.GOOGLE_API_KEY:
        api_key = st.text_input(
            "Enter your Google API Key:",
            type="password"
        )
        st.caption(
            "Get your API key from "
            "[Google AI Studio](https://aistudio.google.com/apikey)"
        )
        if api_key:
            st.session_state.GOOGLE_API_KEY = api_key
            st.success("API key saved")
            st.rerun()
    else:
        st.success("API key configured")
        if st.button("Reset API Key"):
            st.session_state.GOOGLE_API_KEY = None
            st.rerun()

    st.warning(
        "DISCLAIMER:\n\n"
        "This tool is for **educational and informational purposes only**.\n"
        "All outputs must be reviewed by a qualified healthcare professional.\n"
        "Do NOT make medical decisions based solely on this analysis."
    )


# =========================
# MAIN UI
# =========================
st.title("Medical Imaging Diagnosis Agent")
st.write("Upload a medical image for AI-assisted radiological analysis")

if not st.session_state.GOOGLE_API_KEY:
    st.info("Please configure your API key in the sidebar")
    st.stop()

medical_agent = get_medical_agent(st.session_state.GOOGLE_API_KEY)

uploaded_file = st.file_uploader(
    "Upload Medical Image",
    type=["jpg", "jpeg", "png", "dcm"],
    help="Supported formats: JPG, PNG, DICOM"
)

if uploaded_file:
    image = load_medical_image(uploaded_file)
    image = resize_image(image)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Medical Image", use_container_width=True)

    if is_blurry(image):
        st.warning(
            "Image appears blurry. This may reduce analysis confidence."
        )

    analyze_button = st.button(
        "Analyze Image",
        type="primary",
        use_container_width=True
    )

    if analyze_button:
        with st.spinner("Analyzing image..."):
            progress = st.progress(0)

            progress.progress(30)
            agno_image = create_agno_image(image)

            progress.progress(60)
            response: RunOutput = medical_agent.run(
                MEDICAL_ANALYSIS_PROMPT,
                images=[agno_image]
            )

            progress.progress(100)

        st.markdown("## Analysis Results")
        st.markdown(response.content)

        st.caption(
            "AI-generated output. Review by a certified medical professional is required."
        )

else:
    st.info("Upload a medical image to begin")
