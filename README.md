# Deep Learning Playground

Interactive Streamlit app for inference with supplied deep-learning models.

## Live demo

Try the deployed app: [Deep Learning Playground](https://deep-learning-playground-h6vw2q5ztqcjpulmnuemsq.streamlit.app/)

## Features

- **LSTM Sentence Generator:** continue a seed phrase by a selected number of new words.
- **Autoencoder Recommendation System:** generate movie recommendations.
- **Cat vs Dog Classifier:** binary image classification using ResNet18.
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
| Sentence generator | `models/best_lstm_model.keras` | 745-token pre-padded text sequence |
| Tokenizer | `models/tokinizer.pkl` | 8,978-word vocabulary |
| Movie Recommendation | `movie-recommender/sae_model.pth` | 1682-dim rating vector |
| Cat vs Dog Classifier | `models/best_resnet18.pth` | 224x224 RGB Image |

## Before posting or deployment

1. Record a short demo or take screenshots of both pages.
2. Share model artifacts only if you have permission; otherwise document how to place them in `models/`.

## Deployment

The app is built with Streamlit and starts from `app.py`. The root `requirements.txt` contains the runtime dependencies for all three modules.

For deployment, use a Python host that supports TensorFlow, PyTorch, and torchvision and has enough RAM for both frameworks and the supplied model files. Streamlit Community Cloud may not have enough memory for this combination. Configure the host to run `streamlit run app.py`.
