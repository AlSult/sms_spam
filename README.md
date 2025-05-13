# SMS Spam Classifier

A machine learning model to classify SMS messages as spam or ham (not spam) using natural language processing (NLP) techniques.

---

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Process](#modeling-process)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## Project Description

This project builds a classifier to identify spam messages in a dataset of SMS texts. It includes data cleaning, text preprocessing, feature extraction, and model training using popular machine learning algorithms.

---

## Dataset

- **Source**: [UCI - SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Records**: 5,574 SMS messages labeled as `spam` or `ham`.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

To run the full classification pipeline:

```bash
python main.py
```

To use the model for predicting new messages:

```python
from classifier import predict_message

predict_message("Congratulations! You've won a $1000 Walmart gift card. Call now!")
```

---

## Modeling Process

### 1. Data Cleaning
- Removed missing values and duplicates
- Renamed columns for consistency

### 2. Text Preprocessing
- Lowercasing
- Removing punctuation and stopwords
- Lemmatization or stemming
- Tokenization

### 3. Feature Extraction
- TF-IDF Vectorizer or CountVectorizer

### 4. Model Training
- Tested models:
  - Multinomial Naive Bayes (best)
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest

### 5. Evaluation
- Accuracy
- Precision / Recall / F1-score
- Confusion matrix

---

## Results

- **Final Model**: Multinomial Naive Bayes
- **Accuracy**: ~98%
- **Precision (spam)**: ~97%
- **F1-score**: ~97%

---

## Future Work

- Deploy the model using Flask or Streamlit
- Experiment with deep learning (e.g. LSTM)
- Use word embeddings (Word2Vec, GloVe)
- Support multilingual spam detection

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

