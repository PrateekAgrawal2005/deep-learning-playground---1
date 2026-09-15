"""Inference-only Streamlit interface for the supplied deep-learning models."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CAT_DOG_MODEL_PATH = MODELS_DIR / "cat_dog_cnn.keras"
LSTM_MODEL_OPTIONS = {
    "LSTM (30-token context)": MODELS_DIR / "best_lstm_model_30.keras",
    "LSTM (745-token context)": MODELS_DIR / "best_lstm_model.keras",
}
TOKENIZER_PATH = MODELS_DIR / "tokinizer.pkl"
APP_ICON, CAT_ICON, DOG_ICON = "\U0001f9e0", "\U0001f431", "\U0001f436"
PENCIL_ICON, INFO_ICON = "\u270d\ufe0f", "\u2139\ufe0f"
CAT_DOG_PAGE = f"{CAT_ICON} Cat vs Dog Prediction"
SENTENCE_PAGE = f"{PENCIL_ICON} Sentence Generator"
ABOUT_PAGE = f"{INFO_ICON} About"


@st.cache_resource(show_spinner="Loading Cat/Dog model...")
def load_cat_dog_model() -> tf.keras.Model:
    if not CAT_DOG_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Cat/Dog model not found: {CAT_DOG_MODEL_PATH.name}")
    return tf.keras.models.load_model(CAT_DOG_MODEL_PATH, compile=False)


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


def predict_cat_dog(model: tf.keras.Model, image: Image.Image) -> tuple[str, float]:
    """Predict using the CNN's 64x64 RGB, [0, 1]-normalized input."""
    input_shape = model.input_shape
    if not isinstance(input_shape, tuple) or len(input_shape) != 4:
        raise ValueError("The Cat/Dog model must have a 4D image input.")
    height, width, channels = input_shape[1:]
    if not height or not width or channels not in (1, 3):
        raise ValueError(f"Unsupported Cat/Dog input shape: {input_shape}")
    mode = "L" if channels == 1 else "RGB"
    array = np.asarray(image.convert(mode).resize((width, height)), dtype=np.float32)
    if channels == 1:
        array = array[..., np.newaxis]
    output = np.asarray(model.predict(np.expand_dims(array / 255.0, axis=0), verbose=0)).squeeze()
    if np.size(output) != 1:
        raise ValueError("Expected a binary sigmoid output from the Cat/Dog model.")
    dog_probability = float(output)
    label = "DOG" if dog_probability >= 0.5 else "CAT"
    return label, dog_probability if label == "DOG" else 1 - dog_probability


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


def cat_dog_page() -> None:
    st.title(CAT_DOG_PAGE)
    st.write("Upload an image and let the trained CNN model classify it as a cat or dog.")
    uploaded_file = st.file_uploader("Choose a cat or dog image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is None:
        return
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Selected image", use_container_width=True)
    except UnidentifiedImageError:
        st.error("That file is not a valid image.")
        return
    if st.button("Predict", type="primary"):
        try:
            label, confidence = predict_cat_dog(load_cat_dog_model(), image)
            st.subheader("Prediction")
            st.success(f"{CAT_ICON if label == 'CAT' else DOG_ICON} {label}")
            st.metric("Confidence", f"{confidence:.1%}")
        except Exception as error:
            st.error(f"Unable to run the Cat/Dog prediction: {error}")


def sentence_generator_page() -> None:
    st.title(f"{PENCIL_ICON} LSTM Sentence Generator")
    st.write("Enter a starting phrase and generate a continuation using the trained LSTM model.")
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
    st.markdown("- CNN-based Cat vs Dog image classification\n- LSTM-based next-word text generation")
    st.write("The models were trained separately. This application loads the supplied artifacts and performs interactive inference without retraining them.")


def main() -> None:
    st.set_page_config(page_title="Deep Learning Playground", page_icon=APP_ICON, layout="centered")
    st.sidebar.title(f"{APP_ICON} Deep Learning Playground")
    st.sidebar.caption("Interactive model inference")
    page = st.sidebar.radio("Select a Tool", [CAT_DOG_PAGE, SENTENCE_PAGE, ABOUT_PAGE])
    if page == CAT_DOG_PAGE:
        cat_dog_page()
    elif page == SENTENCE_PAGE:
        sentence_generator_page()
    else:
        about_page()


if __name__ == "__main__":
    main()
