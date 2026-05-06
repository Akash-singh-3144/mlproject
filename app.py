from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
application = Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == 'GET':
       return render_template('home.html')
    else:
        