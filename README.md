# Deep Learning Playground

Interactive Streamlit app for inference with supplied deep-learning models.

## Features

- **Cat vs Dog CNN:** upload an image for binary classification and confidence.
- **LSTM Sentence Generator:** continue a seed phrase by a selected number of new words.
- Sidebar navigation, cached model loading, validation, and friendly errors.
- Inference only: no model or tokenizer training is performed.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Models

| Feature | Artifact | Input |
|---|---|---|
| Cat vs Dog classifier | `models/cat_dog_cnn.keras` | 64 x 64 RGB image, normalized to [0, 1] |
| Sentence generator (option 1) | `models/best_lstm_model_30.keras` | 30-token pre-padded text sequence |
| Sentence generator (option 2) | `models/best_lstm_model.keras` | 745-token pre-padded text sequence |
| Tokenizer | `models/tokinizer.pkl` | 8,978-word vocabulary |

## Before posting or deployment

1. Test with one known cat image and one known dog image. The current binary mapping follows the common convention: output below 0.5 is CAT and output at or above 0.5 is DOG. Reverse these labels in `predict_cat_dog()` if your test shows the training labels were opposite.
2. Record a short demo or take screenshots of both pages.
3. Share model artifacts only if you have permission; otherwise document how to place them in `models/`.

## Deployment

Ready for Streamlit Community Cloud or similar Python hosting: push the repository to GitHub, ensure `models/` is included, and use `app.py` as the entry point. TensorFlow hosting needs sufficient memory.
