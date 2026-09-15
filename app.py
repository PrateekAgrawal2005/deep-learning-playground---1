"""Inference-only Streamlit interface for the supplied LSTM models."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
LSTM_MODEL_OPTIONS = {
    "LSTM (30-token context)": MODELS_DIR / "best_lstm_model_30.keras",
    "LSTM (745-token context)": MODELS_DIR / "best_lstm_model.keras",
}
TOKENIZER_PATH = MODELS_DIR / "tokinizer.pkl"

APP_ICON = "\U0001f9e0"
PENCIL_ICON = "\u270d\ufe0f"
INFO_ICON = "\u2139\ufe0f"
SENTENCE_PAGE = f"{PENCIL_ICON} Sentence Generator"
ABOUT_PAGE = f"{INFO_ICON} About"


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


def sentence_generator_page() -> None:
    st.title(f"{PENCIL_ICON} LSTM Sentence Generator")
    st.write("Enter a starting phrase and generate a continuation using a trained LSTM model.")
    seed = st.text_input("Enter your starting text...", placeholder="The important thing is")
    available_models = {
        label: path for label, path in LSTM_MODEL_OPTIONS.items() if path.is_file()
    }
    if not available_models:
        st.error("No LSTM model files were found in the models folder.")
        return
    selected_label = st.selectbox("Choose LSTM model", list(available_models))
    selected_path = available_models[selected_label]
    st.caption(f"Using `{selected_path.name}`")
    count = st.slider("Number of new words to generate", min_value=5, max_value=50, value=15)

    if st.button("Generate continuation", type="primary"):
        if not seed.strip():
            st.warning("Please enter a starting phrase first.")
            return
        try:
            with st.spinner("Generating text..."):
                generated, sequence_length = generate_text(seed, count, selected_path)
            st.subheader("Generated text")
            st.info(generated)
            st.caption(f"Generated exactly {count} new words (max_len = {sequence_length}).")
        except Exception as error:
            st.error(f"Unable to generate text: {error}")


def about_page() -> None:
    st.title(f"{INFO_ICON} About Deep Learning Playground")
    st.write("An inference-only deep learning portfolio project built with Streamlit.")
    st.markdown("- LSTM-based next-word text generation\n- Selectable 30-token and 745-token context models")
    st.write("The models were trained separately. This application loads the supplied artifacts and performs interactive inference without retraining them.")


def main() -> None:
    st.set_page_config(page_title="Deep Learning Playground", page_icon=APP_ICON, layout="centered")
    st.sidebar.title(f"{APP_ICON} Deep Learning Playground")
    st.sidebar.caption("Interactive LSTM inference")
    page = st.sidebar.radio("Select a Tool", [SENTENCE_PAGE, ABOUT_PAGE])
    if page == SENTENCE_PAGE:
        sentence_generator_page()
    else:
        about_page()


if __name__ == "__main__":
    main()
