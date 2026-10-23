# IMDb Review Sentiment Analyzer

An Evolvix AI/ML Internship project that analyzes IMDb movie reviews and predicts whether a review is positive or negative. The app is built with Streamlit and uses a fine-tuned Longformer model.

## Live application

**Deployed URL:** https://evolvix-ai-ml-internship-ahmadtariq2004-cgcv3qnsjx39xt4qsue6s2.streamlit.app/

## Features

- Analyze a movie review and display its predicted sentiment and confidence score.
- Choose between loading the model in the Streamlit app or requesting a prediction from the Task 1 API.
- Handles reviews up to 20,000 characters; in direct model mode, text may be truncated to the model's token limit.
- Includes information about model limitations.

## Model

The app uses the Hugging Face checkpoint [`ahmed792002/Finetuning_Longformer_IMDb_movie_reviews_Classification`](https://huggingface.co/ahmed792002/Finetuning_Longformer_IMDb_movie_reviews_Classification), trained for IMDb movie-review sentiment classification.

## Run locally

1. Make sure Python 3.10 or newer is installed.
2. Open a terminal in the project folder and create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Start the app:

   ```powershell
   streamlit run app.py
   ```

Streamlit will print a local URL to open in your browser. The model is downloaded from Hugging Face the first time you use direct model mode.

## Prediction methods

- **Load model in this app**: Runs the Hugging Face model from the Streamlit app. This is the default method.
- **Call Task 1 API**: Sends the review to the Task 1 prediction API. Configure its URL with the `TASK1_API_URL` environment variable. The API should accept a `POST` request with JSON shaped like `{"text": "..."}` and return a JSON object containing a `sentiment` string and a `score` between `0` and `1`.

For example, in PowerShell:

```powershell
$env:TASK1_API_URL = "http://127.0.0.1:8000/predict"
streamlit run app.py
```

The default API URL is `http://127.0.0.1:8000/predict`. A local URL is only reachable from the same machine; for a hosted app, configure a publicly reachable API endpoint.

## Limitations

Predictions and confidence scores are model estimates, not guarantees of correctness. Sarcasm, mixed opinions, slang, and reviews that differ from the training data can lead to incorrect predictions. Long reviews may be truncated. Do not use the output for high-stakes decisions.
