import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_and_merge_data(fake_path, true_path):
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake['label'] = 0
    true['label'] = 1
    data = pd.concat([fake, true])
    return data[['title', 'text', 'label']]

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_data(data):
    data['text'] = data['text'].apply(clean_text)
    return data

def vectorize_text(data):
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = tfidf.fit_transform(data['text'])
    y = data['label']
    return X, y, tfidf
