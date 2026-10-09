"""Deep Learning Playground — unified Streamlit application.

Pages
-----
🏠 Home             Overview / about the project
🔄 LSTM             Next-word text generation (745-token context model)
🎬 Recommendation   Autoencoder-based movie recommendation system
🐶🐱 Cat vs Dog     Binary image classification using ResNet18
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ──────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# LSTM assets
LSTM_MODEL_OPTIONS = {
    "LSTM (745-token context)": MODELS_DIR / "best_lstm_model.keras",
}
TOKENIZER_PATH = MODELS_DIR / "tokinizer.pkl"

# Movie recommender assets
RECOMMENDER_DIR = BASE_DIR / "movie-recommender"
SAE_MODEL_PATH = RECOMMENDER_DIR / "sae_model.pth"
MOVIES_DAT_PATH = RECOMMENDER_DIR / "data" / "movies.dat"
NB_MOVIES = 1682  # model input/output dimension (MovieLens 100K)

# Cat vs Dog assets
RESNET_MODEL_PATH = MODELS_DIR / "best_resnet18.pth"

# Page labels
APP_ICON = "\U0001f9e0"
HOME_PAGE = "\U0001f3e0 Home"
LSTM_PAGE = "\U0001f504 LSTM"
RECOMMENDER_PAGE = "\U0001f3ac Recommendation System"
CAT_DOG_PAGE = "🐶🐱 Cat vs Dog Classifier"


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM — model loading & inference  (unchanged from existing code)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading LSTM model...")
def load_lstm_model(path_string: str) -> tf.keras.Model:
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"LSTM model not found: {path.name}")
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource(show_spinner="Loading tokenizer...")
def load_tokenizer():
    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH.name}")
    with TOKENIZER_PATH.open("rb") as file:
        return pickle.load(file)


def generate_text(seed_text: str, new_word_count: int, model_path: Path) -> tuple[str, int]:
    model, tokenizer = load_lstm_model(str(model_path)), load_tokenizer()
    sequence_length = model.input_shape[1]
    if not isinstance(sequence_length, int) or sequence_length <= 0:
        raise ValueError(f"Unsupported LSTM input shape: {model.input_shape}")

    words = seed_text.strip().split()
    for _ in range(new_word_count):
        sequence = tokenizer.texts_to_sequences([" ".join(words)])[0][-sequence_length:]
        padded = pad_sequences([sequence], maxlen=sequence_length, padding="pre")
        word_index = np.argmax(model.predict(padded, verbose=0), axis=-1)[0]
        next_word = tokenizer.index_word.get(int(word_index))
        if not next_word:
            raise ValueError("The model predicted a token absent from the tokenizer.")
        words.append(next_word)
    return " ".join(words), sequence_length


# ══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATION SYSTEM — SAE model definition, loading & inference
# ══════════════════════════════════════════════════════════════════════════════
class SAE(nn.Module):
    """Stacked Autoencoder for collaborative filtering.

    Architecture mirrors the training notebook:
        Input (1682) → 20 → 10 → 20 → 1682
    Sigmoid activation is applied after the first three linear layers.
    The final layer produces raw predicted ratings (no activation).
    """

    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(NB_MOVIES, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, NB_MOVIES)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


@st.cache_resource(show_spinner="Loading the trained Autoencoder model…")
def load_sae_model():
    """Recreate the SAE architecture, load pre-trained weights, set to eval mode."""
    if not SAE_MODEL_PATH.is_file():
        st.error(
            f"❌ Model file not found at `{SAE_MODEL_PATH}`.\n\n"
            "Make sure `sae_model.pth` is in the movie-recommender folder."
        )
        st.stop()
    model = SAE()
    model.load_state_dict(
        torch.load(str(SAE_MODEL_PATH), map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


@st.cache_data(show_spinner="Loading movie catalog…")
def load_movies():
    """Load movies.dat and filter to only those the model can score (ID ≤ 1682)."""
    if not MOVIES_DAT_PATH.is_file():
        st.error(
            f"❌ Movie catalog not found at `{MOVIES_DAT_PATH}`.\n\n"
            "Make sure `movies.dat` is inside the `movie-recommender/data/` folder."
        )
        st.stop()
    movies = pd.read_csv(
        MOVIES_DAT_PATH,
        sep="::",
        header=None,
        engine="python",
        encoding="latin-1",
        names=["movie_id", "title", "genres"],
    )
    movies = movies[movies["movie_id"] <= NB_MOVIES].reset_index(drop=True)
    return movies


def get_recommendations(
    model, user_ratings: dict, movies_df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """Run the autoencoder on a new user's ratings and return top-N recommendations."""
    input_vector = np.zeros(NB_MOVIES, dtype=np.float32)
    for movie_id, rating in user_ratings.items():
        input_vector[movie_id - 1] = float(rating)

    input_tensor = torch.FloatTensor(input_vector).unsqueeze(0)

    with torch.no_grad():
        predicted = model(input_tensor)

    predicted_np = predicted.squeeze(0).numpy()

    rated_indices = set(mid - 1 for mid in user_ratings.keys())
    candidates = [
        (idx, predicted_np[idx]) for idx in range(NB_MOVIES) if idx not in rated_indices
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:top_n]

    results = []
    for rank, (idx, pred_rating) in enumerate(top_candidates, start=1):
        movie_id = idx + 1
        movie_row = movies_df[movies_df["movie_id"] == movie_id]
        if not movie_row.empty:
            results.append(
                {
                    "rank": rank,
                    "movie_id": movie_id,
                    "title": movie_row["title"].iloc[0],
                    "genres": movie_row["genres"].iloc[0],
                    "predicted_rating": round(pred_rating, 2),
                }
            )
        else:
            results.append(
                {
                    "rank": rank,
                    "movie_id": movie_id,
                    "title": f"Movie #{movie_id}",
                    "genres": "Unknown",
                    "predicted_rating": round(pred_rating, 2),
                }
            )
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── 🏠 Home ──────────────────────────────────────────────────────────────────
def home_page() -> None:
    st.title(f"{APP_ICON} Deep Learning Playground")
    st.write(
        "An inference-only deep learning portfolio project built with Streamlit."
    )
    st.markdown(
        "---\n"
        "### What's inside\n\n"
        "| Module | Description |\n"
        "|--------|-------------|\n"
        "| 🔄 **LSTM** | Next-word text generation with 745-token context model |\n"
        "| 🎬 **Recommendation System** | Autoencoder-based movie recommendations using a Stacked Autoencoder trained on MovieLens |\n"
        "| 🐶🐱 **Cat vs Dog** | Binary image classification using a custom-trained ResNet18 model |\n"
    )
    st.info(
        "The models were trained separately. This application loads the supplied "
        "artifacts and performs interactive inference without retraining them."
    )


# ── 🔄 LSTM ──────────────────────────────────────────────────────────────────
def lstm_page() -> None:
    st.title("🔄 LSTM Sentence Generator")
    st.write(
        "Enter a starting phrase and generate a continuation using a trained LSTM model."
    )
    seed = st.text_input(
        "Enter your starting text...", placeholder="The important thing is"
    )
    available_models = {
        label: path for label, path in LSTM_MODEL_OPTIONS.items() if path.is_file()
    }
    if not available_models:
        st.error("No LSTM model files were found in the models folder.")
        return
    selected_label = st.selectbox("Choose LSTM model", list(available_models))
    selected_path = available_models[selected_label]
    st.caption(f"Using `{selected_path.name}`")
    count = st.slider(
        "Number of new words to generate", min_value=5, max_value=50, value=15
    )

    if st.button("Generate continuation", type="primary"):
        if not seed.strip():
            st.warning("Please enter a starting phrase first.")
            return
        try:
            with st.spinner("Generating text..."):
                generated, sequence_length = generate_text(seed, count, selected_path)
            st.subheader("Generated text")
            st.info(generated)
            st.caption(
                f"Generated exactly {count} new words (max_len = {sequence_length})."
            )
        except Exception as error:
            st.error(f"Unable to generate text: {error}")


# ── 🎬 Recommendation System ─────────────────────────────────────────────────
def recommender_page() -> None:
    st.title("🎬 Movie Recommendation System")
    st.write(
        "Rate a few movies and get personalized recommendations powered by a "
        "Stacked Autoencoder trained on MovieLens."
    )

    # ── Custom CSS for recommendation cards ──────────────────────────────
    st.markdown(
        """
    <style>
    /* Recommendation card */
    .rec-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        border: 1px solid #3f3f5f;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.85rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .rec-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
    }
    .rec-rank {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #fff;
        font-weight: 700;
        font-size: 0.85rem;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        border-radius: 50%;
        margin-right: 0.6rem;
        vertical-align: middle;
    }
    .rec-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: #e5e7eb;
        vertical-align: middle;
    }
    .rec-genres {
        color: #9ca3af;
        font-size: 0.88rem;
        margin-top: 0.35rem;
        margin-left: 2.4rem;
    }
    .rec-pred {
        margin-top: 0.3rem;
        margin-left: 2.4rem;
    }
    .rec-pred .stars {
        color: #fbbf24;
        font-size: 1rem;
        letter-spacing: 1px;
    }
    .rec-pred .score {
        color: #a78bfa;
        font-weight: 600;
        font-size: 0.9rem;
        margin-left: 0.4rem;
    }
    /* Rated movie tags */
    .rated-tag {
        display: inline-block;
        background: #2d2d44;
        border: 1px solid #4b5563;
        border-radius: 8px;
        padding: 0.35rem 0.75rem;
        margin: 0.2rem 0.25rem;
        font-size: 0.85rem;
        color: #d1d5db;
    }
    .rated-tag .tag-stars {
        color: #fbbf24;
        margin-left: 0.3rem;
    }
    /* Info box */
    .info-box {
        background: #1e293b;
        border-left: 3px solid #667eea;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0.6rem 0;
    }
    /* Pipeline step */
    .pipeline-step {
        text-align: center;
        color: #9ca3af;
        font-size: 0.95rem;
        line-height: 2;
    }
    .pipeline-arrow {
        color: #667eea;
        font-weight: 700;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ── Load model & data ────────────────────────────────────────────────
    sae_model = load_sae_model()
    movies_df = load_movies()

    # ── Session state for user ratings ───────────────────────────────────
    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}

    # ── Rating Section ───────────────────────────────────────────────────
    st.markdown("### 🎯 Rate a few movies to get personalized recommendations")
    st.markdown(
        '<div class="info-box">'
        "💡 Rate at least <strong>3–5 movies</strong> for meaningful recommendations. "
        "The more you rate, the better the model understands your taste."
        "</div>",
        unsafe_allow_html=True,
    )

    movie_options = movies_df["title"].tolist()
    selected_title = st.selectbox(
        "Search or select a movie",
        options=[""] + movie_options,
        index=0,
        placeholder="Type to search…",
        key="movie_selector",
    )

    col_rating, col_add = st.columns([3, 1])
    with col_rating:
        rating_value = st.slider(
            "Your rating",
            min_value=1,
            max_value=5,
            value=4,
            step=1,
            help="1 = Terrible · 2 = Poor · 3 = Average · 4 = Good · 5 = Excellent",
        )
    with col_add:
        st.markdown("<br>", unsafe_allow_html=True)
        add_clicked = st.button("➕ Add Rating", use_container_width=True)

    if add_clicked:
        if selected_title and selected_title != "":
            movie_row = movies_df[movies_df["title"] == selected_title]
            if not movie_row.empty:
                mid = int(movie_row["movie_id"].iloc[0])
                st.session_state.user_ratings[mid] = rating_value
                st.rerun()
            else:
                st.warning("Could not find that movie. Please select from the dropdown.")
        else:
            st.warning("Please select a movie first.")

    # Display current ratings
    if st.session_state.user_ratings:
        st.markdown("#### 🎞️ Your Ratings")
        tags_html = ""
        for mid, rating in st.session_state.user_ratings.items():
            movie_row = movies_df[movies_df["movie_id"] == mid]
            title = (
                movie_row["title"].iloc[0] if not movie_row.empty else f"Movie #{mid}"
            )
            stars = "★" * rating + "☆" * (5 - rating)
            tags_html += (
                f'<span class="rated-tag">'
                f'{title} <span class="tag-stars">{stars}</span>'
                f"</span>"
            )
        st.markdown(tags_html, unsafe_allow_html=True)

        if st.button("🗑️ Clear All Ratings"):
            st.session_state.user_ratings = {}
            st.rerun()
    else:
        st.markdown(
            '<div class="info-box">No movies rated yet. Select a movie above and add your rating.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Recommendation Button ────────────────────────────────────────────
    num_recs = st.radio(
        "Number of recommendations",
        options=[5, 10],
        index=1,
        horizontal=True,
    )

    get_recs = st.button(
        "🚀 Get Recommendations", type="primary", use_container_width=True
    )

    if get_recs:
        if not st.session_state.user_ratings:
            st.error(
                "Please rate at least one movie before requesting recommendations."
            )
        else:
            if len(st.session_state.user_ratings) < 3:
                st.warning(
                    f"You've rated only **{len(st.session_state.user_ratings)}** movie(s). "
                    "While the model can still generate predictions, rating 3–5+ movies "
                    "provides a much richer signal for the autoencoder."
                )
            _show_recommendations(sae_model, movies_df, num_recs)

    # ── How It Works ─────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🔍 How it works"):
        st.markdown(
            """
        This application uses a **Stacked Autoencoder** — a type of neural network —
        trained on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset
        to learn compressed representations of user–movie interactions.

        **Pipeline:**
        """
        )
        st.markdown(
            '<div class="pipeline-step">'
            "📝 Your Ratings "
            '<span class="pipeline-arrow">→</span> '
            "📊 User-Movie Vector (1682-dim) "
            '<span class="pipeline-arrow">→</span> '
            "🔻 Encoder (1682→20→10) "
            '<span class="pipeline-arrow">→</span> '
            "🧠 Latent Representation (10-dim) "
            '<span class="pipeline-arrow">→</span> '
            "🔺 Decoder (10→20→1682) "
            '<span class="pipeline-arrow">→</span> '
            "⭐ Predicted Ratings "
            '<span class="pipeline-arrow">→</span> '
            "🎬 Top Unrated Movies"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
        **Details:**
        - The autoencoder was trained for 200 epochs using MSE loss and RMSprop optimizer.
        - It learns a 10-dimensional latent space that captures movie preference patterns.
        - At inference, your sparse rating vector is compressed and then reconstructed,
          filling in predicted ratings for movies you haven't seen.
        - Movies you've already rated are excluded; the rest are sorted by predicted score.
        """
        )


def _show_recommendations(model, movies_df, top_n):
    """Generate and display recommendation cards."""
    with st.spinner("Running the autoencoder…"):
        results_df = get_recommendations(
            model, st.session_state.user_ratings, movies_df, top_n=top_n
        )

    st.markdown("### 🎯 Recommended For You")
    st.markdown(
        '<div class="info-box">'
        f"Based on your <strong>{len(st.session_state.user_ratings)}</strong> rating(s), "
        "the autoencoder predicted these movies you'd enjoy most."
        "</div>",
        unsafe_allow_html=True,
    )

    for _, row in results_df.iterrows():
        pred_clamped = max(1.0, min(5.0, row["predicted_rating"]))
        full_stars = int(round(pred_clamped))
        stars_str = "★" * full_stars + "☆" * (5 - full_stars)
        genre_parts = row["genres"].replace("|", " · ")

        card_html = f"""
        <div class="rec-card">
            <div>
                <span class="rec-rank">{row['rank']}</span>
                <span class="rec-title">{row['title']}</span>
            </div>
            <div class="rec-genres">{genre_parts}</div>
            <div class="rec-pred">
                <span class="stars">{stars_str}</span>
                <span class="score">{row['predicted_rating']:.2f}</span>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)


# ── 🐶🐱 Cat vs Dog Classifier ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ResNet18 model...")
def load_resnet_model():
    if not RESNET_MODEL_PATH.is_file():
        st.error(f"❌ Model file not found at `{RESNET_MODEL_PATH}`.")
        st.stop()
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(str(RESNET_MODEL_PATH), map_location="cpu", weights_only=True))
    model.eval()
    return model

def cat_dog_page() -> None:
    st.title("🐶🐱 Cat vs Dog Classifier")
    st.write("Upload an image of a cat or a dog, and the ResNet18 model will classify it.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Running inference..."):
                model = load_resnet_model()

                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                input_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.sigmoid(output).item()

                if prob >= 0.5:
                    label = "Dog 🐶"
                    confidence = prob * 100
                else:
                    label = "Cat 🐱"
                    confidence = (1 - prob) * 100

            st.success(f"**Prediction:** {label} (Confidence: {confidence:.2f}%)")
        except Exception as e:
            st.error(f"Error processing image: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — sidebar navigation
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    st.set_page_config(
        page_title="Deep Learning Playground",
        page_icon=APP_ICON,
        layout="centered",
    )
    st.sidebar.title(f"{APP_ICON} Deep Learning Playground")
    st.sidebar.caption("Interactive deep learning inference")

    page = st.sidebar.radio(
        "Navigate",
        [HOME_PAGE, LSTM_PAGE, RECOMMENDER_PAGE, CAT_DOG_PAGE],
    )

    if page == HOME_PAGE:
        home_page()
    elif page == LSTM_PAGE:
        lstm_page()
    elif page == RECOMMENDER_PAGE:
        recommender_page()
    elif page == CAT_DOG_PAGE:
        cat_dog_page()


if __name__ == "__main__":
    main()
