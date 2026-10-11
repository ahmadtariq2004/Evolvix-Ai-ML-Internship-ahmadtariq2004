import math
import os
from html import escape

import requests
import streamlit as st


MODEL_ID = "ahmed792002/Finetuning_Longformer_IMDb_movie_reviews_Classification"
API_URL = os.getenv("TASK1_API_URL", "http://127.0.0.1:8000/predict")
MAX_CHARACTERS = 20000
MAX_TOKENS = 4096

st.set_page_config(
    page_title="IMDb Review Sentiment",
    page_icon="S",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #18372f;
        --muted: #66766e;
        --paper: #f2f3ed;
        --accent: #e66545;
        --lime: #d8f06b;
        --line: #d9e0d8;
    }
    .stApp {
        background:
            linear-gradient(120deg, rgba(220, 229, 215, .38), transparent 46%),
            var(--paper);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1060px; padding: 1.4rem 2rem 3.5rem; }
    .masthead {
        background: #18372f;
        border-radius: 8px;
        color: #f5f6ee;
        margin: 0 0 2rem;
        overflow: hidden;
        padding: 1rem 1.7rem 1.1rem;
        position: relative;
    }
    .masthead-top, .masthead-bottom { align-items: center; display: flex; justify-content: space-between; }
    .brand-lockup { align-items: center; display: flex; gap: .65rem; }
    .brand-mark {
        align-items: center;
        background: var(--lime);
        border-radius: 5px;
        color: var(--ink);
        display: inline-flex;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1rem;
        height: 2rem;
        justify-content: center;
        width: 2rem;
    }
    .brand-name, .masthead-meta, .hero-kicker, .sentiment-key {
        font-size: .75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .brand-name { color: #f5f6ee; }
    .brand-name span { color: var(--lime); }
    .masthead-meta { color: #c6d4c9; }
    .masthead-bottom { align-items: end; margin-top: 1.2rem; position: relative; z-index: 1; }
    .hero-kicker { color: var(--lime); margin-bottom: .55rem; }
    .hero-title {
        color: #f5f6ee;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 2.8rem;
        font-weight: 500;
        line-height: 1.02;
        margin: 0;
    }
    .hero-title em { color: var(--lime); font-weight: 400; }
    .sentiment-key { border-left: 1px solid rgba(245, 246, 238, .25); color: #dce4d8; line-height: 1.9; padding-left: 1.2rem; }
    .sentiment-key span { color: var(--lime); }
    .section-label {
        color: var(--ink);
        font-size: .9rem;
        font-weight: 700;
        margin: .25rem 0 .75rem;
    }
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, .78);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.1rem 1.25rem 1rem;
    }
    [data-testid="stTextArea"] textarea {
        background: #fff;
        border-color: #cbd5cc;
        border-radius: 6px;
        line-height: 1.55;
    }
    [data-testid="stTextArea"] textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
    [data-testid="stFormSubmitButton"] button {
        background: var(--accent);
        border: 1px solid var(--accent);
        color: #fff;
        font-weight: 700;
        min-height: 3rem;
        text-transform: uppercase;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background: #a84425;
        border-color: #a84425;
        color: #fff;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] { gap: .6rem; }
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: rgba(255, 255, 255, .75);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: .45rem .8rem;
    }
    .result-panel {
        background: #fff;
        border: 1px solid var(--line);
        border-left: 5px solid var(--lime);
        border-radius: 8px;
        margin-top: 1.5rem;
        padding: 1.2rem 1.4rem;
    }
    .result-panel.negative { border-left-color: var(--accent); }
    .result-panel.neutral { border-left-color: #4a8d80; }
    .result-heading { color: var(--muted); font-size: .74rem; font-weight: 700; text-transform: uppercase; }
    .result-row { align-items: baseline; display: flex; flex-wrap: wrap; gap: .5rem 1.2rem; justify-content: space-between; }
    .result-value {
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 2.15rem;
        line-height: 1.15;
        margin-top: .25rem;
        overflow-wrap: anywhere;
    }
    .result-score { color: var(--muted); font-size: .9rem; }
    .result-score strong { color: var(--ink); font-size: 1.2rem; margin-left: .35rem; }
    .result-footnote { color: var(--muted); font-size: .82rem; margin-top: .65rem; }
    .stProgress > div > div > div > div { background: var(--accent); }
    .model-caption { color: var(--muted); font-size: .8rem; margin: -.45rem 0 .6rem; overflow-wrap: anywhere; }
    .stAlert { border-radius: 6px; }
    @media (max-width: 640px) {
        .block-container { padding: 1rem 1rem 2.5rem; }
        .masthead { padding: 1rem 1.1rem 1.1rem; }
        .masthead-meta { display: none; }
        .masthead-bottom { align-items: start; margin-top: .9rem; }
        .hero-title { font-size: 2rem; }
        .sentiment-key { display: none; }
        [data-testid="stForm"] { padding: .9rem; }
        .result-value { font-size: 1.75rem; }
    }
    .note {
        border-left: 3px solid var(--accent);
        color: var(--muted);
        font-size: .88rem;
        line-height: 1.55;
        padding: .15rem 0 .15rem .85rem;
    }
    hr { border-color: var(--line); margin: 1.6rem 0 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the sentiment model. The first download may take a few minutes...")
def load_model(model_id):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return tokenizer, model, torch


def predict_with_model(text):
    tokenizer, model, torch = load_model(MODEL_ID)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
    )
    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    labels = model.config.id2label
    label = labels.get(predicted_index, labels.get(str(predicted_index), f"Class {predicted_index}"))
    return label, float(probabilities[predicted_index].item())


def predict_with_api(text):
    response = requests.post(API_URL, json={"text": text}, timeout=(5, 45))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("The API response must be a JSON object.")

    label = payload.get("sentiment")
    score = payload.get("score")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("The API response did not include a valid 'sentiment' label.")
    try:
        score = float(score)
    except (TypeError, ValueError) as error:
        raise ValueError("The API response did not include a valid confidence score.") from error
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("The API confidence score must be between 0 and 1.")
    return label, score


st.markdown(
    '<section class="masthead">'
    '<div class="masthead-top">'
    '<div class="brand-lockup"><span class="brand-mark">E</span>'
    '<span class="brand-name">Evolvix <span>/</span> Task 02</span></div>'
    '<div class="masthead-meta">Cinema / Sentiment</div></div>'
    '<div class="masthead-bottom"><div><div class="hero-kicker">IMDb Review Sentiment</div>'
    '<h1 class="hero-title">Every review<br>has a <em>feeling.</em></h1></div>'
    '<div class="sentiment-key"><span>Positive</span><br>Negative</div></div>'
    '</section>',
    unsafe_allow_html=True,
)

with st.expander("About this model and its limitations"):
    st.write(
        "This app uses the Hugging Face checkpoint trained for IMDb movie-review sentiment classification. "
        "It predicts a label and a model confidence score; it does not understand every nuance of language. "
        "Sarcasm, mixed opinions, slang, and reviews unlike its training data can lead to incorrect results. "
        "The score is not a guarantee of correctness, and long reviews may be truncated. Do not use this output "
        "for high-stakes decisions."
    )

st.markdown('<div class="section-label">Prediction method</div>', unsafe_allow_html=True)
backend = st.radio(
    "Prediction method",
    ("Load model in this app", "Call Task 1 API"),
    horizontal=True,
    label_visibility="collapsed",
    help="Direct model mode is suitable for a hosted app. API mode uses the endpoint configured by TASK1_API_URL.",
)
if backend == "Call Task 1 API":
    st.markdown(f'<div class="model-caption">Task 1 endpoint: {escape(API_URL)}</div>', unsafe_allow_html=True)
    if "127.0.0.1" in API_URL or "localhost" in API_URL:
        st.info("This local endpoint is reachable only from the same machine. For a hosted app, configure a publicly reachable API URL.")
else:
    st.markdown('<div class="model-caption">IMDb-trained sentiment checkpoint</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Analyze a movie review</div>', unsafe_allow_html=True)
with st.form("sentiment_form"):
    review = st.text_area(
        "Movie review",
        placeholder="Paste or write a movie review...",
        height=190,
        max_chars=MAX_CHARACTERS,
        help=f"Up to {MAX_CHARACTERS:,} characters. In direct mode, text beyond the model token limit is truncated.",
    )
    submitted = st.form_submit_button("Analyze review", type="primary", use_container_width=True)

if submitted:
    cleaned_review = review.strip()
    if not cleaned_review:
        st.warning("Enter a movie review before analyzing.")
    else:
        try:
            if backend == "Call Task 1 API":
                with st.spinner("Requesting a prediction from the Task 1 API..."):
                    label, confidence = predict_with_api(cleaned_review)
            else:
                with st.spinner("Analyzing the review..."):
                    label, confidence = predict_with_model(cleaned_review)

            normalized_label = label.casefold()
            result_tone = "positive" if "pos" in normalized_label else "negative" if "neg" in normalized_label else "neutral"
            st.markdown(
                f'<section class="result-panel {result_tone}">'
                '<div class="result-heading">Model prediction</div>'
                '<div class="result-row">'
                f'<div class="result-value">{escape(label)}</div>'
                f'<div class="result-score">Confidence<strong>{confidence:.1%}</strong></div>'
                '</div>'
                '</section>',
                unsafe_allow_html=True,
            )
            st.progress(confidence)
            st.markdown('<div class="result-footnote">Confidence reflects the model output, not a guarantee of correctness.</div>', unsafe_allow_html=True)
        except requests.Timeout:
            st.error("The Task 1 API timed out. Check that it is running and try again.")
        except requests.ConnectionError:
            st.error(f"Could not connect to the Task 1 API at {API_URL}. Start the API or configure TASK1_API_URL.")
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else "unknown"
            st.error(f"The Task 1 API returned HTTP {status}. Check its logs and try again.")
        except requests.RequestException as error:
            st.error(f"The Task 1 API request failed: {error}")
        except ValueError as error:
            st.error(f"Could not use the prediction response: {error}")
        except Exception as error:
            st.error(f"Prediction failed ({type(error).__name__}). Check your network connection, model access, and available memory, then try again.")

st.divider()
st.markdown(
    '<p class="note">Sentiment predictions are estimates, not facts. Review the text and use your own judgment.</p>',
    unsafe_allow_html=True,
)
