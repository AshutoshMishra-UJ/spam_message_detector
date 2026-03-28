import streamlit as st
import pickle
import string
from pathlib import Path
import pandas as pd
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import check_is_fitted

ps = PorterStemmer()


def ensure_nltk_resources():
    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/stopwords": "stopwords",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


ensure_nltk_resources()
STOP_WORDS = set(stopwords.words('english'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in STOP_WORDS and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


def is_fitted(estimator):
    try:
        check_is_fitted(estimator)
        return True
    except (NotFittedError, TypeError, AttributeError):
        return False


def load_training_data(csv_path):
    df = pd.read_csv(csv_path, encoding="latin-1")
    if "v1" in df.columns and "v2" in df.columns:
        label_col, text_col = "v1", "v2"
    else:
        usable_columns = [col for col in df.columns if str(col).strip()]
        if len(usable_columns) < 2:
            raise ValueError("Training data must include at least two columns: label and text.")
        label_col, text_col = usable_columns[0], usable_columns[1]

    labels = df[label_col].astype(str).str.lower().str.strip().map({"ham": 0, "spam": 1})
    texts = df[text_col].astype(str)

    valid_rows = labels.notna() & texts.notna()
    cleaned_texts = texts[valid_rows].apply(transform_text)
    cleaned_labels = labels[valid_rows].astype(int)

    if cleaned_texts.empty:
        raise ValueError("No valid rows found in training data.")

    return cleaned_texts, cleaned_labels


def load_or_train_artifacts(base_dir):
    vectorizer_path = base_dir / "vectorizer.pkl"
    model_path = base_dir / "model.pkl"
    data_path = base_dir / "spam.csv"

    tfidf = None
    model = None

    if vectorizer_path.exists():
        with open(vectorizer_path, "rb") as f:
            tfidf = pickle.load(f)

    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    if tfidf is None or not hasattr(tfidf, "transform"):
        tfidf = TfidfVectorizer(max_features=3000)

    if model is None or not hasattr(model, "predict"):
        model = MultinomialNB()

    vectorizer_ready = is_fitted(tfidf)
    model_ready = is_fitted(model)
    retrained = False

    if not vectorizer_ready or not model_ready:
        texts, labels = load_training_data(data_path)

        if vectorizer_ready:
            features = tfidf.transform(texts)
        else:
            features = tfidf.fit_transform(texts)

        model = MultinomialNB()
        model.fit(features, labels)

        with open(vectorizer_path, "wb") as f:
            pickle.dump(tfidf, f)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        retrained = True

    return tfidf, model, retrained


BASE_DIR = Path(__file__).resolve().parent
tfidf, model, artifacts_retrained = load_or_train_artifacts(BASE_DIR)

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 15%, #fff1dc 0%, transparent 45%),
            radial-gradient(circle at 85% 10%, #f0f8ff 0%, transparent 40%),
            linear-gradient(160deg, #f9fafb 0%, #eef5ff 100%);
    }

    .hero {
        background: linear-gradient(120deg, #1f4b99 0%, #2674d8 55%, #35b5a2 100%);
        border-radius: 18px;
        padding: 26px 24px;
        color: #ffffff;
        box-shadow: 0 18px 30px rgba(15, 43, 84, 0.24);
        margin-bottom: 18px;
    }

    .hero h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
    }

    .hero p {
        margin: 10px 0 0 0;
        font-size: 0.98rem;
        opacity: 0.95;
    }

    .section-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(31, 75, 153, 0.13);
        border-radius: 16px;
        padding: 16px 16px 12px 16px;
        box-shadow: 0 8px 20px rgba(15, 43, 84, 0.08);
    }

    .result-card {
        border-radius: 14px;
        padding: 14px 16px;
        margin-top: 14px;
        font-weight: 600;
        border: 1px solid transparent;
        animation: fadeIn 0.35s ease;
    }

    .spam {
        background: #fff2ef;
        color: #ba2b1b;
        border-color: #ffc4bd;
    }

    .safe {
        background: #ebfff8;
        color: #066f4d;
        border-color: #9ee7cd;
    }

    .hint {
        color: #355070;
        font-size: 0.92rem;
        margin-bottom: 0.35rem;
    }

    .stButton > button {
        border-radius: 999px;
        border: none;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        color: #ffffff;
        background: linear-gradient(100deg, #0f4ca6 0%, #1f77e5 100%);
        box-shadow: 0 8px 18px rgba(31, 119, 229, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 20px rgba(31, 119, 229, 0.36);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>📩 Smart SMS Spam Detector</h1>
        <p>Paste a message below and get an instant spam vs. safe prediction from your trained NLP model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### How it works")
    st.write("1. Text is cleaned and stemmed")
    st.write("2. Message is vectorized using TF-IDF")
    st.write("3. Trained model predicts spam probability")
    if artifacts_retrained:
        st.success("Model artifacts were automatically retrained from spam.csv.")
    st.markdown("---")
    st.caption("Tip: Include full SMS content for better accuracy.")

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<p class="hint">Enter an SMS or email-like short message to classify.</p>', unsafe_allow_html=True)
input_sms = st.text_area("Message", placeholder="Example: Congratulations! You won a free ticket. Click now to claim.", height=160)

predict_clicked = st.button('Predict Message')
st.markdown('</div>', unsafe_allow_html=True)

if predict_clicked:
    if not input_sms.strip():
        st.warning("Please enter a message first.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(model.predict_proba(vector_input)[0][int(result)])

        if result == 1:
            st.markdown('<div class="result-card spam">⚠️ Prediction: Spam</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-card safe">✅ Prediction: Not Spam</div>', unsafe_allow_html=True)

        if confidence is not None:
            st.progress(confidence)
            st.caption(f"Model confidence: {confidence * 100:.2f}%")
