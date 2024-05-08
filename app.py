import json
from flask import Flask,request,app,jsonify,url_for,render_template
from flask_cors import CORS
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
from joblib import load

app=Flask(__name__)
CORS(app)
# Load the JobLib files
symptoms_list = load('symptoms_file.gz')
loaded_svm_model = load('svm_model_file.gz')
df_specialist = load('specialist_file.gz')

def find_specialist(disease):
    specialist = df_specialist.loc[df_specialist['Disease'] == disease, 'Specialist'].values
    if len(specialist) > 0:
        return f"{str(specialist[0])}"
    else:
        return "No Specialist"

def format_disease_response(prediction_dict):
  formatted_response = defaultdict(list)
  disease_count = 1
  for disease, val in prediction_dict.items():
    val.insert(0,disease)
    formatted_response[f'Disease-{disease_count}'].extend(val)
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
        # print(user_symptoms)
        if not user_symptoms:
            return jsonify({'error': 'Missing symptom data'}), 400

        user_input_vector = [0 for _ in range(len(symptoms_list))]
        for val in user_symptoms:
            if val in symptoms_list:
                user_input_vector[symptoms_list.index(val)] = 1

        # Predict top 3 diseases
        predicted_probabilities = loaded_svm_model.predict_proba([user_input_vector])[0]

        # Sort the probabilities
        top_3_indices = sorted(range(len(predicted_probabilities)), key=lambda i: predicted_probabilities[i], reverse=True)[:3]

        # Extract top 3 diseases and their probabilities
        top_3_diseases = [loaded_svm_model.classes_[i].strip().title() for i in top_3_indices]
        top_3_probabilities = [predicted_probabilities[i] for i in top_3_indices]

        top3_dict = defaultdict(list)
        for disease, probability in zip(top_3_diseases, top_3_probabilities):
            specialist = find_specialist(disease)
            top3_dict[disease].extend([round(probability * 100, 2), specialist])
        
        print(top3_dict)
        formatted_response = format_disease_response(top3_dict)

        return jsonify(formatted_response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__=="__main__":
    app.run(debug=True)