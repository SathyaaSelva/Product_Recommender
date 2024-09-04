from flask import Flask,send_from_directory, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open('product_recommendation_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_id = data.get('user_id')
    item_id = data.get('item_id')

    if not user_id or not item_id:
        return jsonify({'error': 'Invalid input'}), 400

    try:
        # Perform prediction
        prediction = model.predict(user_id, item_id)

        # Check if prediction is valid
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # Return rounded prediction
        return jsonify({'prediction': round(prediction.est, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
