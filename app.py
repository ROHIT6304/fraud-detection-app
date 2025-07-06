from flask import Flask, render_template, request
import pickle
import numpy as np
from geopy.distance import geodesic

app = Flask(__name__)

# Load saved model and encoders
model = pickle.load(open("fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    cc_num = int(data['cc_num'])
    amt = float(data['amt'])
    merchant = label_encoders["merchant"].transform([data["merchant"]])[0]
    category = label_encoders["category"].transform([data["category"]])[0]

    # Calculate distance
    user_loc = (float(data['lat']), float(data['long']))
    merch_loc = (float(data['merch_lat']), float(data['merch_long']))
    distance = geodesic(user_loc, merch_loc).km

    # Build feature vector and scale
    features = np.array([[cc_num, merchant, category, amt, distance]])
    scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(scaled)[0]
    result = "🔍 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
