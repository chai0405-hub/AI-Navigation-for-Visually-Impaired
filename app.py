"""
AI Navigation for Visually Impaired — Streamlit app.

Run locally:   streamlit run app.py
"""

import streamlit as st
from PIL import Image

from navigator import (APPROX_METERS, build_guidance, detect, load_model,
                       speak)

st.set_page_config(
    page_title="AI Navigation for Visually Impaired",
    page_icon="🦯",
    layout="centered",
)

# Large, high-contrast styling for low-vision users.
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 20px; }
    .guidance {
        font-size: 1.6rem; font-weight: 700; line-height: 1.45;
        padding: 1rem 1.2rem; border-radius: 12px; margin: .6rem 0 1rem;
        border: 3px solid currentColor;
    }
    .stop    { background:#3b0000; color:#ffdddd; }
    .turn    { background:#3a2a00; color:#ffe9a8; }
    .careful { background:#2b2b00; color:#fff7a8; }
    .clear   { background:#00301a; color:#c8ffe0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading AI model (first run takes ~30 seconds)…")
def get_model():
    return load_model("yolov8n.pt")


# ----------------------------- Sidebar settings -----------------------------
LANGS = {"English": "en", "हिन्दी (Hindi)": "hi", "ಕನ್ನಡ (Kannada)": "kn"}
with st.sidebar:
    st.header("⚙️ Settings")
    lang_label = st.selectbox("Voice language", list(LANGS.keys()))
    lang = LANGS[lang_label]
    voice_on = st.toggle("Speak guidance aloud", value=True)
    conf = st.slider("Detection sensitivity (confidence)", 0.2, 0.8, 0.4, 0.05,
                     help="Lower = detects more objects, but more mistakes.")
    max_items = st.slider("Objects to announce", 1, 8, 4)
    show_image = st.toggle("Show annotated image", value=True)
    st.caption("Distances are rough estimates from object size in the photo, "
               "not measurements. Always use a cane or guide as the primary aid.")

# --------------------------------- Main UI ---------------------------------
st.title("🦯 AI Navigation for Visually Impaired")
st.write("Point the phone camera in the direction you are walking and take a photo. "
         "The app detects obstacles, tells you where they are, and suggests which way to move.")

tab_cam, tab_upload = st.tabs(["📷 Camera", "🖼️ Upload image"])
with tab_cam:
    cam = st.camera_input("Take a photo of the path ahead")
with tab_upload:
    up = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])

source = cam or up
if source is None:
    st.info("Waiting for a photo… Tip: hold the phone at chest height, facing forward.")
    st.stop()

image = Image.open(source)
model = get_model()

with st.spinner("Analysing the scene…"):
    detections, annotated = detect(model, image, conf=conf)
    text, decision = build_guidance(detections, lang=lang, max_items=max_items)

css = {"stop": "stop", "go_left": "turn", "go_right": "turn",
       "careful": "careful", "clear": "clear"}[decision]
st.markdown(f'<div class="guidance {css}" role="alert" aria-live="assertive">{text}</div>',
            unsafe_allow_html=True)

if voice_on:
    try:
        st.audio(speak(text, lang), format="audio/mp3", autoplay=True)
    except Exception as e:  # gTTS needs internet access
        st.warning(f"Voice output unavailable right now ({e}). Showing text only.")

if show_image:
    st.image(annotated, caption="Detected objects", width="stretch")

if detections:
    st.subheader("Details")
    st.dataframe(
        [{
            "Object": d.name,
            "Position": {"left": "Left", "center": "Ahead", "right": "Right"}[d.position],
            "Approx. distance": APPROX_METERS[d.distance],
            "In walking path": "Yes" if d.in_path else "No",
            "Confidence": f"{d.confidence:.0%}",
        } for d in detections],
        width="stretch", hide_index=True,
    )

st.caption("Built with Streamlit, YOLOv8 (Ultralytics) and gTTS.")
