# Spam Message Detector

An end-to-end NLP project to classify SMS messages as Spam or Not Spam using classic machine learning models and a Streamlit frontend.

## Project Overview

This project includes:

- Data preprocessing and feature engineering in notebook form
- Multiple model experiments (accuracy and precision comparison)
- Saved artifacts for inference (`model.pkl`, `vectorizer.pkl`)
- A production-style Streamlit app (`app.py`) with improved UI
- Auto-recovery in app startup if model artifacts are missing or not fitted

## Folder Structure

```text
.
|-- app.py
|-- model.pkl
|-- vectorizer.pkl
|-- spam.csv
|-- sms-spam-detection.ipynb
|-- requirements.txt
|-- setup.sh
|-- Procfile
|-- nltk.txt
`-- README.md
```

## Requirements

### System Requirements

- Python 3.10 or newer
- pip
- Git (optional, for cloning)

### Python Dependencies

Install from `requirements.txt`:

- streamlit
- nltk
- scikit-learn
- pandas

## Model Performance

The following metrics were reported in `sms-spam-detection.ipynb` on the project test split.

| Model | Accuracy | Precision |
| --- | ---: | ---: |
| SVC | 0.975822 | 0.974790 |
| KNN (KN) | 0.905222 | 1.000000 |
| Naive Bayes (NB) | 0.970986 | 1.000000 |
| Decision Tree (DT) | 0.929400 | 0.828283 |
| Logistic Regression (LR) | 0.956480 | 0.969697 |
| Random Forest (RF) | 0.976789 | 0.975000 |
| Extra Trees (ETC) | 0.977756 | 0.967480 |
| Bagging (BgC) | 0.959381 | 0.869231 |
| AdaBoost | 0.923598 | 0.839080 |
| Gradient Boosting (GBDT) | 0.950677 | 0.930693 |
| XGBoost (xgb) | 0.970986 | 0.950000 |

Notes:

- Precision is critical in spam detection because false positives can hide real user messages.
- `model.pkl` used by `app.py` is currently based on Multinomial Naive Bayes in this repository code path.

## How To Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/AshutoshMishra-UJ/spam_message_detector.git
cd spam_message_detector
```

### 2. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open the URL shown in terminal (usually `http://localhost:8501`).

## App Behavior Details

- On startup, app ensures required NLTK resources are available (`punkt`, `stopwords`).
- If `model.pkl` or `vectorizer.pkl` is missing or not fitted, app automatically retrains from `spam.csv` and saves fresh artifacts.
- Input text is normalized, tokenized, stopwords removed, stemmed, then transformed with TF-IDF before prediction.

## Reproducing Training Workflow

To reproduce feature engineering and model experimentation, open and run:

- `sms-spam-detection.ipynb`

This notebook contains:

- Data cleaning and transformations
- Vectorization and training
- Accuracy/precision benchmarking across multiple models

## Troubleshooting

### `NotFittedError` in app

If you see `NotFittedError`, the latest app code auto-retrains artifacts from `spam.csv` on startup.
If needed, delete existing artifacts and restart:

```bash
rm model.pkl vectorizer.pkl
streamlit run app.py
```

On Windows PowerShell:

```powershell
Remove-Item model.pkl, vectorizer.pkl
streamlit run app.py
```

### NLTK resource lookup errors

The app downloads required corpora/tokenizers automatically. If your network blocks this, manually install NLTK data in your environment.

## Future Improvements

- Add recall, F1-score, ROC-AUC comparison table in README
- Add automated training script (`train.py`) with artifact versioning
- Add unit tests and CI checks for model loading and prediction paths
