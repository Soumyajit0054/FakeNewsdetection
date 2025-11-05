from flask import Flask, render_template,request,flash,redirect,url_for
import pandas as pd
import pickle
import re, string
import os

app = Flask(__name__)


# Load model and vectorizer using paths relative to this script to avoid working-directory issues.
def _load_pickle(path, name):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    raise FileNotFoundError(
        f"Required file '{name}' not found at {path}.\n"
        "Place the file there or update the path in the code."
    )

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model.pkl')
vectorizer_path = os.path.join(script_dir, 'vectorizer.pkl')

model = _load_pickle(model_path, 'model.pkl')
vectorizer = _load_pickle(vectorizer_path, 'vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    # Use raw string literals for regex patterns so backslashes are interpreted by the regex engine,
    # not by Python string escaping (avoids SyntaxWarning).
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    return text

@app.route('/data')
def data():
	df = pd.read_csv(r'c:\Users\User\Downloads\fake_news_dataset.csv')
	return df.head().to_html()


@app.route('/')
def home():
	
	return render_template('home.html')

@app.route('/contactus')
def contactus():
	
	return render_template('contact_us.html')

@app.route('/aboutus')
def aboutus():
	# Render the home page (templates/home.html)
	return render_template('about.html')

@app.route('/services')
def services():
	# Render the home page (templates/home.html)
	return render_template('services.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get input from form
#         news_text = request.form['newsInput']

#         # Clean and vectorize input
#         cleaned_text = clean_text(news_text)
#         vect_text = vectorizer.transform([cleaned_text])

#         # Predict
#         prediction = model.predict(vect_text)[0]

#         if prediction == 1:
#             result = "The news seems to be REAL!"
#         else:
#             result = "The news seems to be FAKE!"
        
#         return render_template('services.html', prediction=result)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form.get('newsInput', '').strip()

        if not news_text:
            return render_template('services.html', prediction="⚠️ Please enter some news text to analyze.")

        # Clean and vectorize
        cleaned_text = clean_text(news_text)
        vect_text = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(vect_text)[0]
        proba = model.predict_proba(vect_text)[0]

        fake_prob = proba[0]  # Probability of being fake
        real_prob = proba[1]  # Probability of being real
        confidence = max(proba) * 100  # Convert to %

        # Custom logic for interpretation
        if prediction == 1 and real_prob >= 0.7:
            result = f"✅ The news seems to be REAL (Confidence: {real_prob*100:.2f}%)"
        elif prediction == 0 and fake_prob >= 0.7:
            result = f"🚨 The news seems to be FAKE (Confidence: {fake_prob*100:.2f}%)"
        else:
            result = f"⚠️ The news is likely to be FAKE (Low confidence: {confidence:.2f}%)"

        # Pass both the prediction and the original text back to the template
        return render_template('services.html', prediction=result, news_text=news_text)



if __name__ == '__main__':
	# Run the development server for quick local testing
	app.run(host='127.0.0.1', port=5000, debug=True)

