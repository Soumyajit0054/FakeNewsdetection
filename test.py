import numpy as np
import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.is_trained = False

    # ---------------------------
    # 1️⃣ TEXT CLEANING FUNCTION
    # ---------------------------
    def clean_text(self, text):
        """Cleans and preprocesses text data"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
        text = re.sub(r'\w*\d\w*', '', text)  # Remove words with numbers
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text

    # ---------------------------
    # 2️⃣ LOAD AND CLEAN DATA
    # ---------------------------
    def load_and_preprocess_data(self, file_path):
        """Load and clean the dataset"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at: {file_path}")

        print(f"📂 Loading dataset from: {file_path}")
        data = pd.read_csv(file_path)
        print(f"Initial data shape: {data.shape}")

        # Handle missing values
        data['title'] = data['title'].fillna('')
        data['text'] = data['text'].fillna('')

        # Combine title + text
        data['content'] = data['title'] + ' ' + data['text']

        # Convert label to numeric (0 = Fake, 1 = Real)
        data['label'] = data['label'].astype(str).str.lower().str.strip()
        data['label'] = data['label'].map({'fake': 0, 'real': 1})

        # Drop rows with invalid labels
        data = data.dropna(subset=['label'])

        # Clean text
        print("🧹 Cleaning text data...")
        data['content'] = data['content'].apply(self.clean_text)
        data = data[data['content'].str.strip() != '']

        print(f"✅ Final data shape: {data.shape}")
        print(f"Label distribution:\n{data['label'].value_counts()}")
        return data

    # ---------------------------
    # 3️⃣ TRAIN MODELS
    # ---------------------------
    def train_models(self, file_path, test_size=0.2, random_state=42):
        """Train and evaluate multiple models"""
        data = self.load_and_preprocess_data(file_path)
        X = data['content']
        y = data['label']

        # TF-IDF Vectorization
        print("⚙️ Vectorizing text data...")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=5000)
        X_vectorized = self.vectorizer.fit_transform(X)

        # Split into train-test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            "Multinomial NB": MultinomialNB()
        }

        best_score = 0
        best_model = None
        best_name = ""

        print("\n==============================")
        print("🔍 TRAINING MODELS")
        print("==============================")

        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {acc:.4f}")

            if acc > best_score:
                best_score = acc
                best_model = model
                best_name = name

        self.model = best_model
        self.is_trained = True

        # Final evaluation
        y_pred = self.model.predict(X_test)
        print(f"\n🏆 Best Model: {best_name} (Accuracy: {best_score:.4f})")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))
        print("\n📈 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return self.model, self.vectorizer

    # ---------------------------
    # 4️⃣ PREDICTION FUNCTION
    # ---------------------------
    def predict_news(self, text):
        """Predict if a news text is Fake or Real"""
        if not self.is_trained:
            raise ValueError("Model not trained. Please train or load the model first.")

        cleaned = self.clean_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        pred = self.model.predict(vectorized)[0]
        proba = self.model.predict_proba(vectorized)[0]

        return {
            'text': text,
            'prediction': 'FAKE' if pred == 0 else 'REAL',
            'confidence': max(proba),
            'fake_probability': proba[0],
            'real_probability': proba[1]
        }

    # ---------------------------
    # 5️⃣ SAVE / LOAD MODEL
    # ---------------------------
    def save_model(self, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("✅ Model and vectorizer saved successfully!")

    def load_model(self, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        self.is_trained = True
        print("✅ Model and vectorizer loaded successfully!")


# ---------------------------
# 🚀 MAIN FUNCTION
# ---------------------------
def main():
    detector = FakeNewsDetector()
    dataset_path = 'fake_news_dataset.csv'

    # Train model (you can comment this after first run)
    model, vectorizer = detector.train_models(dataset_path)
    detector.save_model()

    # Load model (for later use)
    detector.load_model()

    # Test predictions
    test_texts = [
        "Breaking: Scientists discover revolutionary cure for all diseases!",
        "The Prime Minister addressed the nation on the new budget today.",
        "Aliens have landed in Times Square according to sources.",
        "The weather forecast predicts light rain this weekend."
    ]

    print("\n==============================")
    print("📰 FAKE NEWS DETECTION RESULTS")
    print("==============================")

    for t in test_texts:
        result = detector.predict_news(t)
        print(f"\nText: {t[:80]}...")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
        print(f"Fake: {result['fake_probability']:.2%}, Real: {result['real_probability']:.2%}")


if __name__ == "__main__":
    main()
