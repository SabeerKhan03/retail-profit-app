# app.py
# This is your Flask API — the "door" to your model.
# It listens for incoming data, runs the prediction, and sends back the result.
# Run it with: python app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd

# ── Start Flask ────────────
app = Flask(__name__)
# Flask(__name__) just means "create a web server for this file"

# ── Load model files (runs once when server starts)
print("Loading model...")
model           = joblib.load('retail_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')
meta            = joblib.load('model_meta.pkl')
print("Model loaded! Server ready.")

# Route 1: Health check
# A route is a URL path. This one is just to confirm the server is running.
# Visit http://localhost:5000/health in browser to test.
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Server is running!'})

#  Route 2: Get dropdown options 
# Streamlit calls this to know what options to show in dropdowns
@app.route('/meta', methods=['GET'])
def get_meta():
    return jsonify(meta)

#  Route 3: Make a prediction 
# This is the main route. Streamlit sends order data here,
# the model predicts profit/loss, and we send back the result.
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data Streamlit sent
        data = request.get_json()
        print("Received data:", data)   # helpful for debugging

        # Build a single-row table from the input
        # (same structure as your training data)
        raw = pd.DataFrame([{
            'Sales':         float(data['Sales']),
            'Discount':      float(data['Discount']),  # already decimal (0.2 not 20%)
            'Delivery Days': int(data['Delivery Days']),
            'Sub-Category':  data['Sub-Category'],
            'Region':        data['Region'],
            'Segment':       data['Segment'],
            'Ship Mode':     data['Ship Mode'],
        }])

        # Encode categories to numbers (same as training)
        raw_encoded = pd.get_dummies(raw, drop_first=True).astype(int)

        # CRITICAL STEP: Align columns to exactly match training
        # If a category wasn't seen before, its column won't exist
        # reindex adds missing columns as 0 (= not present)
        raw_encoded = raw_encoded.reindex(
            columns=feature_columns, fill_value=0
        )

        # Run prediction
        predicted_profit = model.predict(raw_encoded)[0]
        label = "Profit" if predicted_profit >= 0 else "Loss"

        return jsonify({
            'predicted_profit': round(float(predicted_profit), 2),
            'label': label,
        })

    except Exception as e:
        # If anything goes wrong, send back the error message
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 400


# Start the server
if __name__ == '__main__':
    # host='0.0.0.0' means: accept connections from anywhere (needed for AWS later)
    # port=5000 means: listen on port 5000
    # debug=False means: don't auto-reload (important for stability)
    app.run(host='0.0.0.0', port=5000, debug=False)