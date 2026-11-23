# Fake News Detection using NLP & Machine Learning

An educational text-classification project that predicts whether a news headline/article is **Fake** or **Real** using text preprocessing, TF-IDF features, and machine-learning classifiers.

> **Disclaimer:** This is an educational classifier, not a fact-checking system. Predictions should not be treated as proof of truth or falsity.

## Project Overview
- Clean and preprocess news text
- Convert text into numerical features using TF-IDF
- Train and compare machine-learning models
- Evaluate using accuracy, precision, recall, F1-score and a confusion matrix
- Make predictions through a Streamlit interface

## Dataset
The scripts expect a CSV named `News_dataset.csv` with `title` and `label` columns. Label convention used in the original notebook:
- `0`: Fake
- `1`: Real

**Dataset is not included** in this repository. Download it from the source you used and place it in the project root. Follow the dataset provider's license/terms.

## Repository Structure
```
├── Fake News Classification.ipynb
├── train_model.py
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── News_dataset.csv  (local only; not committed)
```

## Setup
1. Clone:
   ```bash
   git clone https://github.com/jeetpatilprofessional-blip/Fake-News-Detection-NLP-ML.git
   cd Fake-News-Detection-NLP-ML
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
   Activate it on Windows:
   ```bash
   .venv\\Scripts\\activate
   ```
   macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place `News_dataset.csv` in the project root.
5. Train and evaluate:
   ```bash
   python train_model.py
   ```
6. Launch the app:
   ```bash
   streamlit run app.py
   ```

## Methodology
1. Load the dataset and select title and label fields.
2. Remove missing records and duplicate titles.
3. Split into train and test sets **before fitting vectorization**.
4. Use TF-IDF within a scikit-learn Pipeline to avoid test-data leakage.
5. Compare Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
6. Evaluate the models and save the best pipeline.

## Evaluation
The training script prints accuracy, precision, recall, F1-score, classification report, and confusion matrix. Actual results are generated when run on your dataset; no performance numbers are claimed in advance.

## Limitations
- The model learns patterns in its dataset and may not generalize to other sources.
- Headlines may not contain sufficient context.
- Satire, bias, language variation, and evolving events complicate classification.
- A prediction is not a substitute for checking trustworthy sources.

## Tech Stack
Python, Pandas, scikit-learn, Streamlit.

## Author
Jeet Patil
