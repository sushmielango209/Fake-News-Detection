# Fake-News-Detection

## 🔍 Overview
Fake news spreads misinformation and undermines public trust. This project uses Natural Language Processing (NLP) and machine learning to classify news articles as **real** or **fake**, contributing to digital literacy and social awareness.

## 🎯 Objective
Build a binary classifier that flags fake news using text data and TF-IDF features.

## 📁 Dataset
Source: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download)

- **Fake.csv**: 23,502 fake news articles  
- **True.csv**: 21,417 real news articles  
- **Columns**: `title`, `text`, `subject`, `date`

## ⚙️ Project Structure

```
FakeNewsDetector/
├── data/                  # Raw CSV files
├── models/                # Saved ML models
├── notebooks/             # Optional Jupyter exploration
├── src/                   # Modular Python scripts
│   ├── preprocess.py      # Text cleaning & TF-IDF
│   ├── train.py           # Model training
│   └── evaluate.py        # Accuracy & F1 reporting
├── main.py                # Pipeline entry point
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## 🧪 Installation & Execution

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/FakeNewsDetector.git
cd FakeNewsDetector
```

### 2. Create virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python main.py
```

## 🧼 Preprocessing

- Lowercasing
- Removing punctuation
- Removing stopwords
- TF-IDF vectorization

## 🧠 Models Used

- **Passive Aggressive Classifier** (default)
- **Support Vector Machine** (optional)

Switch model in `main.py`:
```python
train_model(X, y, model_type='svm')  # or 'passive'
```

## 📊 Evaluation Metrics

- **Accuracy**
- **F1 Score**
- **Classification Report**

Sample output:
```
Accuracy: 0.93
F1 Score: 0.92
```
