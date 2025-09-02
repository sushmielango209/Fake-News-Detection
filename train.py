from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

def train_model(X, y, model_type='passive'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'passive':
        model = PassiveAggressiveClassifier(max_iter=50)
    else:
        model = SVC(kernel='linear')
    
    model.fit(X_train, y_train)
    joblib.dump(model, f'models/{model_type}_model.pkl')
    return model, X_test, y_test
