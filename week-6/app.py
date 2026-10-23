"""Streamlit dashboard for drawing or uploading a digit for prediction."""

from __future__ import annotations

from io import BytesIO
import os

import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps


APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(__file__), "digit_cnn.keras")
H5_MODEL_PATH = os.path.join(APP_DIR, "digit_cnn.h5")
WEIGHTS_PATH = os.path.join(APP_DIR, "digit_cnn.weights.h5")

try:
    import tensorflow as tf
except Exception as exc:
    tf = None
    TENSORFLOW_ERROR = str(exc)
else:
    TENSORFLOW_ERROR = ""

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None


st.set_page_config(
    page_title="Scribble2Num | AI Number Scanner",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
        --bg: #080d18;
        --surface: rgba(19, 28, 47, .72);
        --surface-strong: rgba(23, 34, 56, .9);
        --line: rgba(148, 163, 184, .18);
        --text: #edf2ff;
        --muted: #9aa8c1;
        --indigo: #6366f1;
        --cyan: #06b6d4;
        --mint: #78f0dc;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 14% 0%, rgba(99, 102, 241, .20), transparent 34rem),
            radial-gradient(ellipse at 92% 22%, rgba(6, 182, 212, .12), transparent 30rem),
            linear-gradient(145deg, #080d18 0%, #0b1220 54%, #0d1424 100%);
        color: var(--text);
        font-family: 'Manrope', sans-serif;
    }
    h1, h2, h3, h4 { color: var(--text); font-family: 'Manrope', sans-serif; }
    .block-container { max-width: 1480px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, .98), rgba(9, 14, 25, .98));
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: var(--text); }
    [data-testid="stSidebar"] .react-aria-ComboBox [role="group"] {
        background: rgba(17, 27, 46, .95); border: 1px solid var(--line);
        border-radius: 10px; min-height: 2.7rem;
    }
    [data-testid="stSidebar"] .react-aria-ComboBox input[role="combobox"] {
        background: transparent; color: var(--text);
    }
    [data-testid="stSidebar"] .react-aria-ComboBox button {
        background: transparent; border: 0; color: var(--muted);
    }
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
    .hero-wrap { text-align: center; padding: .5rem 0 1.2rem; }
    .hero-eyebrow {
        color: var(--cyan); font-family: 'DM Mono', monospace; font-size: .72rem;
        letter-spacing: .18em; text-transform: uppercase; margin-bottom: .8rem;
    }
    .hero-title {
        background: linear-gradient(100deg, #f8fbff 12%, #9ca6ff 48%, #58e4ed 88%);
        background-clip: text; -webkit-background-clip: text; color: transparent;
        font-size: clamp(2.8rem, 6vw, 5rem); font-weight: 800;
        line-height: 1.02; margin: 0 0 .65rem;
    }
    .hero-subtitle { color: var(--muted); font-size: 1rem; letter-spacing: .02em; }
    .hero-status { display: flex; justify-content: center; margin: .9rem 0 1.5rem; }
    .section-kicker {
        color: var(--cyan); font-family: 'DM Mono', monospace; font-size: .72rem;
        letter-spacing: .12em; text-transform: uppercase; margin: .25rem 0 .8rem;
    }
    .glass-panel, div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 16px; box-shadow: 0 18px 48px rgba(0, 0, 0, .24);
        backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    }
    div[data-testid="stMetric"] {
        background: rgba(18, 28, 47, .76); border: 1px solid var(--line);
        border-radius: 12px; padding: .85rem 1rem;
    }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] { color: var(--text); }
    .result-banner {
        background: linear-gradient(125deg, rgba(99, 102, 241, .2), rgba(6, 182, 212, .13)),
                    rgba(12, 19, 33, .94);
        border: 1px solid rgba(111, 173, 255, .34); border-radius: 18px;
        box-shadow: 0 0 36px rgba(99, 102, 241, .16), inset 0 1px rgba(255,255,255,.08);
        padding: 1.35rem 1.5rem; margin: .2rem 0 1rem;
    }
    .result-label { color: #b1c0d8; font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .result-number {
        color: #f3f8ff; display: block; font-family: 'DM Mono', monospace;
        font-size: 3rem; font-weight: 500; line-height: 1.2; margin: .2rem 0 .65rem;
        overflow-wrap: anywhere; text-shadow: 0 0 24px rgba(6, 182, 212, .38);
    }
    .summary-row { display: flex; flex-wrap: wrap; gap: .5rem; }
    .summary-chip {
        background: rgba(6, 182, 212, .1); border: 1px solid rgba(6, 182, 212, .25);
        border-radius: 999px; color: #a8f3f4; font-size: .75rem; font-weight: 700;
        padding: .4rem .7rem;
    }
    .mini-card { background: rgba(19, 28, 47, .72); border: 1px solid var(--line); border-radius: 14px; padding: .65rem; }
    .preview-frame img, [data-testid="stImage"] img { border: 1px solid var(--line); border-radius: 13px; }
    iframe { border: 1px solid rgba(148, 163, 184, .25); border-radius: 14px; }
    div[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, var(--indigo), var(--cyan)); }
    div.stButton > button {
        background: linear-gradient(105deg, var(--indigo), #4f8cf6 52%, var(--cyan));
        border: 0; border-radius: 10px; color: #fff; font-weight: 700;
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
    }
    div.stButton > button:hover { box-shadow: 0 8px 24px rgba(6, 182, 212, .22); filter: brightness(1.08); transform: translateY(-2px); }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stExpander"] { background: rgba(19, 28, 47, .58); border: 1px solid var(--line); border-radius: 14px; }
    [data-testid="stFileUploader"] section { background: rgba(19, 28, 47, .56); border: 1px dashed rgba(99, 102, 241, .48); border-radius: 14px; }
    [data-testid="stRadio"] label { border-radius: 10px; }
    .stCode { border-radius: 10px; }
    @media (max-width: 760px) {
        .block-container { padding: 1.2rem 1rem 2rem; }
        .hero-title { font-size: 2.8rem; }
        .result-number { font-size: 2.25rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the digit model...")
def load_model(model_path: str):
    if tf is None:
        return None

    model_load_error = None
    if os.path.isfile(model_path):
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as exc:
            model_load_error = exc

    if os.path.isfile(H5_MODEL_PATH):
        try:
            return tf.keras.models.load_model(H5_MODEL_PATH, compile=False)
        except Exception as exc:
            model_load_error = exc

    if os.path.isfile(WEIGHTS_PATH):
        try:
            fallback_model = build_fallback_model()
            fallback_model.load_weights(WEIGHTS_PATH)
            return fallback_model
        except Exception as exc:
            model_load_error = exc

    if model_load_error is not None:
        raise ValueError(
            "Could not load the .keras model or its HDF5 fallbacks. "
            f"Last loading error: {model_load_error}"
        ) from model_load_error
    raise FileNotFoundError(
        "No model artifact found. Expected digit_cnn.keras, digit_cnn.h5, "
        "or digit_cnn.weights.h5 beside app.py."
    )


def build_fallback_model():
    """Recreate the training CNN so legacy HDF5 weights can be loaded."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1), name="input_layer"),
            tf.keras.layers.Rescaling(1.0 / 255, name="rescaling"),
            tf.keras.layers.Conv2D(
                32, kernel_size=3, padding="same", activation="relu", name="conv2d"
            ),
            tf.keras.layers.MaxPooling2D(pool_size=2, name="max_pooling2d"),
            tf.keras.layers.Conv2D(
                64, kernel_size=3, padding="same", activation="relu", name="conv2d_1"
            ),
            tf.keras.layers.MaxPooling2D(pool_size=2, name="max_pooling2d_1"),
            tf.keras.layers.Dropout(0.25, name="dropout"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(128, activation="relu", name="dense"),
            tf.keras.layers.Dropout(0.4, name="dropout_1"),
            tf.keras.layers.Dense(10, activation="softmax", name="dense_1"),
        ],
        name="handwritten_digit_cnn",
    )


def image_to_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert PIL or canvas data to RGB, compositing transparent pixels on white."""
    if isinstance(image, Image.Image):
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    else:
        pixels = np.asarray(image, dtype=np.uint8)
        if pixels.ndim == 2:
            return cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
        if pixels.shape[2] == 3:
            return pixels.copy()
        rgba = pixels

    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(rgb * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)


def segment_digits(image: Image.Image | np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]], np.ndarray]:
    """Find left-to-right digit contours and create normalized 28x28 ROIs."""
    rgb = image_to_rgb(image)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = [
        cv2.boundingRect(contour)
        for contour in contours
        if cv2.boundingRect(contour)[2] >= 5 and cv2.boundingRect(contour)[3] >= 10
    ]
    boxes.sort(key=lambda box: box[0])

    digit_images = []
    for x, y, width, height in boxes:
        roi = binary[y : y + height, x : x + width]
        square_size = max(width, height) + 20
        square = np.zeros((square_size, square_size), dtype=np.uint8)
        offset_x = (square_size - width) // 2
        offset_y = (square_size - height) // 2
        square[offset_y : offset_y + height, offset_x : offset_x + width] = roi
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        digit_images.append(resized.astype(np.float32) / 255.0)

    if digit_images:
        digit_batch = np.stack(digit_images, axis=0)[..., np.newaxis]
    else:
        digit_batch = np.empty((0, 28, 28, 1), dtype=np.float32)
    return rgb, binary, boxes, digit_batch


def predict_digits(model, digit_batch: np.ndarray) -> np.ndarray:
    """Predict a batch while honoring the model's built-in Rescaling layer."""
    model_batch = digit_batch
    first_layer = model.layers[0] if model.layers else None
    if tf is not None and isinstance(first_layer, tf.keras.layers.Rescaling):
        scale = np.asarray(first_layer.scale, dtype=np.float32)
        offset = np.asarray(first_layer.offset, dtype=np.float32)
        if scale.size == 1 and not np.isclose(float(scale), 0):
            model_batch = (digit_batch - offset) / scale
    return np.asarray(model.predict(model_batch, verbose=0), dtype=np.float32)


def annotate_predictions(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    digits: np.ndarray,
    confidences: np.ndarray,
) -> np.ndarray:
    """Draw green boxes and per-digit labels onto an RGB image."""
    annotated = image.copy()
    green = (44, 145, 82)
    for (x, y, width, height), digit, confidence in zip(boxes, digits, confidences):
        cv2.rectangle(annotated, (x, y), (x + width, y + height), green, 2)
        label = f"{int(digit)}  {confidence:.0%}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_top = max(0, y - text_height - baseline - 6)
        cv2.rectangle(
            annotated,
            (x, label_top),
            (x + text_width + 8, label_top + text_height + baseline + 6),
            green,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x + 4, label_top + text_height + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def render_multidigit_results(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    digit_batch: np.ndarray,
) -> None:
    if not boxes:
        st.warning("No digits detected. Write clear digits with a little space between them.")
        return
    if model is None:
        st.info("The digits are segmented. A trained model is needed to predict them.")
        return

    probabilities = predict_digits(model, digit_batch)
    digits = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    number = "".join(str(int(digit)) for digit in digits)
    average_confidence = float(confidences.mean())

    st.markdown(
        f'<div class="result-banner"><span class="result-label">Scribble2Num result</span>'
        f'<strong class="result-number">{number}</strong>'
        f'<div class="summary-row"><span class="summary-chip">{len(digits)} DIGITS</span>'
        f'<span class="summary-chip">AVG CONFIDENCE {average_confidence:.1%}</span>'
        f'<span class="summary-chip">READ LEFT TO RIGHT</span></div></div>',
        unsafe_allow_html=True,
    )
    st.code(number, language=None)
    st.markdown('<div class="section-kicker">Annotated vision preview</div>', unsafe_allow_html=True)
    st.image(annotate_predictions(image, boxes, digits, confidences), width="stretch")

    with st.expander("🔍 Segmented Digit Breakdown", expanded=True):
        for start in range(0, len(digits), 4):
            columns = st.columns(min(4, len(digits) - start))
            for column, index in zip(columns, range(start, min(start + 4, len(digits)))):
                with column:
                    with st.container(border=True):
                        st.image(
                            np.round(digit_batch[index, ..., 0] * 255).astype(np.uint8),
                            caption=f"Position {index + 1} · 28 × 28",
                            width=96,
                        )
                        st.badge(
                            f"Digit {int(digits[index])}",
                            icon=":material/tag:",
                            color="blue",
                        )
                        st.progress(
                            float(confidences[index]),
                            text=f"Confidence · {confidences[index]:.1%}",
                        )


MODEL_ARTIFACTS = (MODEL_PATH, H5_MODEL_PATH, WEIGHTS_PATH)

if tf is not None and any(os.path.isfile(path) for path in MODEL_ARTIFACTS):
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Could not load `{MODEL_PATH.name}`: {exc}")
        model = None
else:
    model = None

if model is not None:
    model_status = "Model online"
    status_color = "green"
    status_icon = ":material/check_circle:"
    status_detail = "CNN ready · 10 digit classes"
elif tf is None:
    model_status = "Runtime needs attention"
    status_color = "orange"
    status_icon = ":material/warning:"
    status_detail = "TensorFlow could not load"
else:
    model_status = "Training required"
    status_color = "blue"
    status_icon = ":material/model_training:"
    status_detail = "Local dataset detected · 10 classes"

with st.sidebar:
    st.markdown("## ⚙️ Studio Controls")
    mode = st.radio(
        "Input mode",
        ["🎨 Interactive Sketchpad", "📁 Upload Image"],
        label_visibility="collapsed",
        key="scribble_input_mode",
    )
    stroke_width = st.slider("Brush stroke width", min_value=10, max_value=30, value=18, step=2)
    ink_presets = {
        "Graphite": "#111827",
        "Indigo ink": "#3730a3",
        "Deep cyan": "#155e75",
    }
    ink_name = st.selectbox("Ink color", list(ink_presets))
    background_presets = {
        "Bright white": "#ffffff",
        "Cool paper": "#f1f5f9",
    }
    background_name = st.selectbox("Canvas background", list(background_presets))
    with st.expander("ℹ️ About Scribble2Num"):
        st.write(
            "OpenCV converts the image to a clean binary mask, finds and orders digit contours, "
            "then the CNN classifies each 28 × 28 crop. The predictions are joined into one number."
        )
        st.caption("OpenCV contour segmentation · TensorFlow CNN · local inference")
    if tf is None:
        with st.expander("TensorFlow setup details"):
            st.warning("TensorFlow could not be imported. Check that TensorFlow and NumPy versions are compatible.")
            st.code(TENSORFLOW_ERROR or "TensorFlow import failed.")
    elif model is None:
        st.info("Train the local CNN with `python train.py` to enable predictions.")

st.markdown(
    '<div class="hero-wrap"><div class="hero-eyebrow">AI NUMBER SCANNER &nbsp;·&nbsp; COMPUTER VISION</div>'
    '<div class="hero-title">Scribble2Num</div>'
    '<div class="hero-subtitle">Interactive Multi-Digit Recognition &amp; Vision Pipeline</div></div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="hero-status">', unsafe_allow_html=True)
st.badge(model_status, icon=status_icon, color=status_color)
st.markdown("</div>", unsafe_allow_html=True)

input_image = None
input_boxes = None
input_batch = None
is_blank_canvas = False

input_column, output_column = st.columns([1.04, 0.96], gap="large", vertical_alignment="top")
with input_column:
    st.markdown('<div class="section-kicker">01 &nbsp; / &nbsp; INPUT STUDIO</div>', unsafe_allow_html=True)
    with st.container(border=True):
        if mode == "🎨 Interactive Sketchpad":
            st.markdown("### Interactive sketchpad")
            st.caption("Write a complete number from left to right.")
            st.info("Tip: Leave a small gap between digits for accurate segmentation.", icon=":material/lightbulb:")
            canvas = None
            if st_canvas is None:
                st.error("The drawing canvas is unavailable. Use the image upload mode instead.")
            else:
                canvas = st_canvas(
                    fill_color="rgba(0, 0, 0, 0)",
                    stroke_width=stroke_width,
                    stroke_color=ink_presets[ink_name],
                    background_color=background_presets[background_name],
                    height=200,
                    width=600,
                    drawing_mode="freedraw",
                    display_toolbar=True,
                    key="scribble2num_canvas",
                )
            if canvas is not None and canvas.image_data is not None:
                canvas_rgb, binary, boxes, digit_batch = segment_digits(canvas.image_data)
                if cv2.countNonZero(binary) == 0:
                    is_blank_canvas = True
                else:
                    input_image, input_boxes, input_batch = canvas_rgb, boxes, digit_batch
        else:
            st.markdown("### Upload an image")
            st.caption("Choose a clear image containing one handwritten number.")
            uploaded_file = st.file_uploader(
                "Select a number image",
                type=["png", "jpg", "jpeg", "bmp", "webp"],
                help="Digits are segmented, read left-to-right, and classified individually.",
            )
            if uploaded_file is not None:
                uploaded_image = ImageOps.exif_transpose(
                    Image.open(BytesIO(uploaded_file.getvalue()))
                ).convert("RGB")
                st.image(uploaded_image, caption=uploaded_file.name, width="stretch")
                input_image, _, input_boxes, input_batch = segment_digits(uploaded_image)

with output_column:
    st.markdown('<div class="section-kicker">02 &nbsp; / &nbsp; LIVE AI OUTPUT</div>', unsafe_allow_html=True)
    if input_image is not None:
        render_multidigit_results(input_image, input_boxes, input_batch)
    elif is_blank_canvas:
        with st.container(border=True):
            st.markdown("### Your result will appear here")
            st.caption("Add one or more digits to the sketchpad to begin.")
    elif mode == "📁 Upload Image":
        with st.container(border=True):
            st.markdown("### Your result will appear here")
            st.caption("Upload an image to see the number, contour boxes, and digit confidence.")
    else:
        with st.container(border=True):
            st.markdown("### Your result will appear here")
            st.caption("Segmented digits and confidence analytics will appear here.")

st.markdown(
    '<div style="border-top:1px solid rgba(148,163,184,.16); margin-top:2rem; padding-top:.9rem; '
    'color:#9aa8c1; font-family:\'DM Mono\',monospace; font-size:.7rem; letter-spacing:.08em">'
    'SCRIBBLE2NUM &nbsp;·&nbsp; LOCAL INFERENCE &nbsp;·&nbsp; OPENCV + CNN</div>',
    unsafe_allow_html=True,
)