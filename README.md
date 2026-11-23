# Fake News Detection using NLP & Machine Learning

An educational text-classification project that predicts whether a news headline/article is **Fake** or **Real** using text preprocessing, TF-IDF features, and machine-learning classifiers.

> **Disclaimer:** This is an educational classifier, not a fact-checking system. Predictions should not be treated as proof of truth or falsity.

## Features
- Text cleaning and TF-IDF vectorization
- Comparison of Logistic Regression, Multinomial Naive Bayes, and Linear SVM
- Evaluation using accuracy, precision, recall, F1-score, classification report, and confusion matrix
- Streamlit interface for interactive classification
- Linear SVM decision-margin display (explicitly not presented as probability)
- Separate transparent text-pattern signals for urgency, sharing pressure, alarmist language, and similar patterns
- Verification checklist to encourage checking claims independently

## Dataset
The training script expects a CSV named `News_dataset.csv` with `title` and `label` columns. Label convention used in the project:
- `0`: Fake
- `1`: Real

Dataset is not included. Obtain it from your original source, place it in the project root, and follow its license/terms.

## Setup
1. Clone:
   ```bash
   git clone https://github.com/jeetpatilprofessional-blip/Fake-News-Detection-NLP-ML.git
   cd Fake-News-Detection-NLP-ML
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   ```
   Windows:
   ```bash
   .venv\\Scripts\\activate
   ```
   macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```
3. Install:
   ```bash
   pip install -r requirements.txt
   ```
4. Place `News_dataset.csv` in the project root.
5. Train:
   ```bash
   python train_model.py
   ```
6. Launch:
   ```bash
   streamlit run app.py
   ```

## Methodology
1. Load and select title and label fields.
2. Remove missing rows and duplicate titles.
3. Split into train and test sets before fitting vectorization.
4. Fit TF-IDF inside a scikit-learn pipeline.
5. Compare Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
6. Save the best pipeline by weighted F1.

## Interpreting the app
- **Model prediction:** The output of the trained classifier.
- **Decision margin:** Signed distance-like score returned by Linear SVM; it is not a probability and is not calibrated confidence.
- **Text-pattern signals:** Simple keyword/phrase patterns shown separately. They do not change the ML prediction and do not establish truth.
- **Verification checklist:** Suggested manual steps for checking claims.

## Limitations
- Learns dataset-specific language patterns, not truth.
- Headlines may not contain enough context.
- False positives and false negatives are possible.
- The heuristic signals are simplistic and may flag legitimate content or miss misinformation.
- No live source checking or fact verification is performed.

## Tech Stack
Python, Pandas, scikit-learn, Streamlit, Joblib.

## Author
Jeet Patil
