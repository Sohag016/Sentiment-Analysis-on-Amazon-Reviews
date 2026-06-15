# 🛍️ Sentiment Analysis on Amazon Product Reviews

> Predicting the sentiment of Amazon product reviews using classical Machine Learning models with TF-IDF vectorization, hyperparameter tuning, and comparative evaluation.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Objective](#-objective)
3. [Dataset](#-dataset)
4. [Project Workflow](#-project-workflow)
   - [Section 1 — Dataset Overview](#section-1--dataset-overview)
   - [Section 2 — Data Preprocessing](#section-2--data-preprocessing)
   - [Section 3 — Model Selection](#section-3--model-selection)
   - [Section 4 — Model Training](#section-4--model-training)
   - [Section 5 — Formal Evaluation](#section-5--formal-evaluation)
   - [Section 6 — Hyperparameter Tuning](#section-6--hyperparameter-tuning)
   - [Section 7 — Comparative Analysis](#section-7--comparative-analysis)
   - [Section 8 — Conclusion & Comments](#section-8--conclusion--comments)
5. [Models Used](#-models-used)
6. [Evaluation Metrics](#-evaluation-metrics)
7. [Installation & Setup](#-installation--setup)
8. [How to Run](#-how-to-run)
9. [Results](#-results)
10. [Technologies Used](#-technologies-used)

---

## 🔍 Project Overview

This project performs **binary sentiment classification** on Amazon product reviews. Each review is labeled as either **Positive (1)** or **Negative (0)**. The pipeline covers end-to-end NLP processing — from raw text cleaning and vectorization to training, evaluating, and comparing multiple machine learning classifiers.

---

## 🎯 Objective

- Predict the **sentiment** (Positive / Negative) of an Amazon product review from its text content (`reviewText`)
- Compare the performance of **5 classical ML models** using TF-IDF features
- Optimize top models via **hyperparameter tuning** (GridSearchCV)
- Identify the **best-performing model** and visualize results

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | Public GitHub Dataset (rashakil-ds) |
| **URL** | https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/amazon.csv |
| **Text Column** | `reviewText` |
| **Label Column** | `Positive` (1 = Positive, 0 = Negative) |
| **Task** | Binary Sentiment Classification |

> The dataset is loaded directly from the URL — no manual download required.

---

## 🔄 Pipeline Architecture

```
Amazon Reviews Dataset (CSV)
          │
          ▼
  Drop Missing Values
          │
          ▼
  Text Preprocessing
  (Lowercase → Remove Punctuation
   → Tokenize → Remove Stopwords → Lemmatize)
          │
          ▼
  Train-Test Split (80/20)
          │
          ▼
  TF-IDF Vectorization (max 5000 features)
          │
          ▼
  Train 5 ML Models
  (LR · RF · SVM · Naïve Bayes · XGBoost)
          │
          ▼
  Evaluate (Accuracy, Precision, Recall, F1)
          │
          ▼
  Hyperparameter Tuning (GridSearchCV)
          │
          ▼
  Comparative Analysis + Confusion Matrix
          │
          ▼
  Best Model Identified & Summarized
```

---

## 📁 Project Workflow

### Section 1 — Dataset Overview
Loads the Amazon reviews CSV directly from a public URL using `pandas`. The dataset contains two key columns: `reviewText` (raw review text) and `Positive` (binary sentiment label). Displays the first few rows to inspect the data structure.

### Section 2 — Data Preprocessing
Cleans and prepares the raw text data through the following steps:

- **Drop missing values** — Removes rows where `reviewText` or `Positive` is null
- **Lowercase conversion** — Standardizes all text to lowercase
- **Punctuation removal** — Strips all special characters and punctuation
- **Tokenization** — Splits text into individual word tokens using NLTK
- **Stopword removal** — Filters out common English stopwords (e.g., "the", "is", "and")
- **Lemmatization** — Reduces words to their base dictionary form using `WordNetLemmatizer`
- **Train-Test Split** — 80% training / 20% testing with `random_state=42`
- **TF-IDF Vectorization** — Converts cleaned text to a numeric matrix with top 5000 features

### Section 3 — Model Selection
Initializes **5 machine learning models** for comparative evaluation:

| Model | Library |
|---|---|
| Logistic Regression | `sklearn.linear_model` |
| Random Forest | `sklearn.ensemble` |
| Support Vector Machine (SVM) | `sklearn.svm.LinearSVC` |
| Naïve Bayes | `sklearn.naive_bayes.MultinomialNB` |
| XGBoost | `xgboost.XGBClassifier` |

### Section 4 — Model Training
Trains all 5 models on the TF-IDF-transformed training set. Each model is fitted using `model.fit(X_train_tfidf, y_train)` and stored for evaluation. Training progress is printed for each model.

### Section 5 — Formal Evaluation
Evaluates all trained models on the test set using 4 metrics:

- **Accuracy** — Overall correct predictions
- **Precision** — Of predicted positives, how many are truly positive
- **Recall** — Of actual positives, how many were correctly identified
- **F1 Score** — Harmonic mean of precision and recall

Full `classification_report` is printed for each model.

### Section 6 — Hyperparameter Tuning
Applies **GridSearchCV** (3-fold cross-validation) to find the best hyperparameters for:

- **Random Forest:** Searches over `n_estimators` ∈ {50, 100, 200} and `max_depth` ∈ {10, 20, None}
- **Logistic Regression:** Searches over regularization parameter `C` ∈ {0.1, 1, 10}

Best parameters are printed after each search.

### Section 7 — Comparative Analysis
Summarizes and compares all model results in a single table. Plots a grouped **bar chart** (Accuracy, Precision, Recall, F1) for all 5 models side-by-side to visually identify the best performer.

### Section 8 — Conclusion & Comments
- Automatically identifies the **best model** by highest accuracy using `idxmax()`
- Plots the **Confusion Matrix** (Seaborn heatmap) for the best model showing True/False Positives and Negatives
- Prints a final summary: Best model name, Accuracy, Precision, Recall, F1 Score
- Discusses challenges, lessons learned, and key findings

---

## 🤖 Models Used

| Model | Key Hyperparameters |
|---|---|
| **Logistic Regression** | `max_iter=1000`, `C` tuned via GridSearch |
| **Random Forest** | `n_estimators=100`, `max_depth` tuned via GridSearch |
| **Linear SVM** | `LinearSVC`, `random_state=42` |
| **Naïve Bayes** | `MultinomialNB` (default) |
| **XGBoost** | `eval_metric='logloss'`, `random_state=42` |

---

## 📏 Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | (TP + TN) / Total predictions |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) |
| **Confusion Matrix** | Visual breakdown of predictions vs. true labels |

---

## ⚙️ Installation & Setup

### Requirements
- Python 3.8+
- Google Colab or local Jupyter environment

### Install Dependencies

```bash
pip install pandas scikit-learn xgboost nltk seaborn matplotlib
```

### Download NLTK Resources

The notebook downloads these automatically, or run manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## ▶️ How to Run

1. **Open the notebook** — Upload `Sentiment_Analysis_on_Amazon_Product_Reviews.ipynb` to Google Colab or Jupyter.

2. **Run all cells** — The dataset loads automatically from the public URL; no file upload needed.

3. **View results** — Each section prints evaluation metrics, comparison tables, and plots inline.

---

## 📈 Results

The best-performing model is determined automatically at runtime. All 5 models are compared across Accuracy, Precision, Recall, and F1 Score. A confusion matrix is generated for the best model showing per-class classification performance on the test set.

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **ML Library** | scikit-learn |
| **Boosting** | XGBoost |
| **NLP** | NLTK (tokenization, stopwords, lemmatization) |
| **Vectorization** | TF-IDF (`TfidfVectorizer`) |
| **Tuning** | GridSearchCV (3-fold CV) |
| **Visualization** | matplotlib, seaborn |
| **Data Handling** | pandas |
| **Environment** | Google Colab / Jupyter Notebook |

## 🙋‍♂️ Author:
**Name:** [MD SOHAG HOSSAIN]  
**Role:** Machine Learning Intern @CodeAlpha  
**Date:** August 2025  
