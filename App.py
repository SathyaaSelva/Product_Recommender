from flask import Flask, send_from_directory, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
with open('product_recommendation_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    product = data.get('key_product_name')

    if not product:
        return jsonify({'error': 'Invalid input, product name is required'}), 400

    try:
        # Since model.predict requires both user_id and item_id, we'll use product name for both.
        prediction = model.predict(product, product)

        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # Return rounded prediction
        return jsonify({'prediction': round(prediction.est, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
