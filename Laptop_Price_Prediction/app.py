from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np


model = joblib.load('laptop_price_model.pkl')
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    try:
    
        data = request.json
        print("Received data:", data)     
        features = pd.DataFrame({
    'Company': [data['Company']],
    'TypeName': [data['TypeName']],
    'Ram': [int(data['Ram'])],
    'OpSys': [data['OpSys']],
    'IPS': [int(data['IPS'])],
    'Touchscreen': [int(data['Touchscreen'])],
    'Resolution_Type': [data['Resolution_Type']],
    'Storage_Type': [data['Storage_Type']],
    'GPU_Brand': [data['GPU_Brand']],
    'CPU_Brand': [data['CPU_Brand']],
    'Memory_GB': [int(data['Memory_GB'])],
    'Weight': [float(data['Weight'])],
    'Inches': [float(data['Inches'])]
})
        prediction = model.predict(features)

        return jsonify({'prediction': float(np.round(prediction[0], 2))})
    except Exception as e:
        
        print("Error during prediction:", e)
        return jsonify({'error': str(e)}), 400
@app.route('/contact')
def contact():
    return render_template('contact.html')
if __name__ == '__main__':
    app.run(debug=True)
