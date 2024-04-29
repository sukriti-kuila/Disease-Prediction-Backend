import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
from flask_cors import CORS
import numpy as np
import pandas as pd

app=Flask(__name__)
CORS(app)
# Load the PKL files
symptoms_list = pickle.load(open('Symptoms_List.pkl','rb'))
# classifier_model=pickle.load(open('classifier.pkl','rb'))
# scalar=pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def api_home():
    return "<h3 style=\"text-align: center\">WELCOME TO MEDICO APP</h3>"

@app.route('/medico/api/symptoms')
def get_symptoms():
    return jsonify(symptoms_list)

if __name__=="__main__":
    app.run(debug=True)