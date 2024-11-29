from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and encoder
model = joblib.load('iris_logistic_regression_model.pkl')
encoder = joblib.load('iris_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Renders the index.html template

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input data from frontend (JSON)
#         data = request.get_json()
#         input_data = np.array(data['input'], dtype=np.float64).reshape(1, -1)
        
#         # Make prediction
#         prediction = model.predict(input_data)
        
#         # Convert prediction to native Python types
#         prediction = int(prediction[0])  # Convert numpy int64 to native Python int
        
#         # Get the flower class label using the encoder
#         class_name = encoder.inverse_transform([prediction])[0]
        
#         # Return prediction as JSON response
#         return jsonify({'prediction': class_name})
    
#     except Exception as e:
#         return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
