import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
from flask_cors import CORS
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
from joblib import load

app=Flask(__name__)
CORS(app)
# Load the PKL/JobLib files
symptoms_list = pickle.load(open('symptoms_list_pkl.pkl','rb'))
loaded_rf_model = load('rf_model_file.gz')

def format_disease_response(prediction_dict):
  formatted_response = OrderedDict()
  disease_count = 1
  for disease, probability in prediction_dict.items():
    formatted_response[f'Disease-{disease_count}'] = [disease, probability]
    disease_count += 1

  return formatted_response

@app.route('/')
def api_home():
    return "<h3 style=\"text-align: center\">WELCOME TO MEDICO APP</h3>"

@app.route('/api/predict/disease',methods = ['POST'])
def predict_disease():
    try:
        req = request.get_json()
        user_symptoms = req['data']
        print(user_symptoms)
        if not user_symptoms:
            return jsonify({'error': 'Missing symptom data'}), 400

        user_input_vector = [0 for _ in range(len(symptoms_list))]
        for val in user_symptoms:
            if val in symptoms_list:
                user_input_vector[symptoms_list.index(val)] = 1

        # Predict top 3 diseases based on the input symptoms and their probabilities
        predicted_probabilities = loaded_rf_model.predict_proba([user_input_vector])[0]

        # Sort the probabilities
        top_3_indices = sorted(range(len(predicted_probabilities)), key=lambda i: predicted_probabilities[i], reverse=True)[:3]

        # Extract top 3 diseases and their probabilities
        top_3_diseases = [loaded_rf_model.classes_[i].strip().title() for i in top_3_indices]
        top_3_probabilities = [predicted_probabilities[i] for i in top_3_indices]


        top3_dict = defaultdict(float)
        for disease, probability in zip(top_3_diseases, top_3_probabilities):
            top3_dict[disease] = round(probability * 100, 2)

        print(top3_dict)
        formatted_response = format_disease_response(top3_dict)

        return jsonify(formatted_response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__=="__main__":
    app.run(debug=True)