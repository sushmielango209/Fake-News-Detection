from src.preprocess import load_and_merge_data, preprocess_data, vectorize_text
from src.train import train_model
from src.evaluate import evaluate_model

data = load_and_merge_data('data/Fake.csv', 'data/True.csv')
data = preprocess_data(data)
X, y, tfidf = vectorize_text(data)
model, X_test, y_test = train_model(X, y, model_type='passive')
evaluate_model(model, X_test, y_test)
