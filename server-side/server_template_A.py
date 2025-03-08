from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import requests
from requests.exceptions import ConnectionError
from urllib3.exceptions import MaxRetryError, NameResolutionError
import json
from datetime import datetime as dt

# ff. imports are for getting secret values from .env file
from pathlib import Path
import os

from modelling.utilities.loaders import load_model
from modelling.utilities.preprocessors import (
    translate_labels,
    encode_image,
    standardize_image,
    activate_logits,
    decode_one_hot,
    translate_labels,
    re_encode_sparse_labels
)
from modelling.utilities.visualizers import (
    show_image
)
import numpy as np

from PIL import Image

# configure location of build file and the static html template file
app = Flask(__name__, template_folder='static')

# since simple html from url http://127.0.0.1:5000 requests to
# api endpoint at http://127.0.0.1:5000/ we must set the allowed
# origins or web apps with specific urls like http://127.0.0.1:5000
# to be included otherwise it will be blocked by CORS policy
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5000",])

model_names = []
models = []
scaler = None
encoder = None

def load_models():
    """
    prepares and loads sample input and custom model in
    order to use trained weights/parameters/coefficients
    """

    # recreate model architecture
    saved_lgbm = load_model('./modelling/saved/models/lgbm.pkl')
    model_names.append(type(saved_lgbm).__name__)
    models.append(saved_lgbm)

def load_preprocessors():
    """
    prepares and loads the saved encoders, normalizers of
    the dataset to later transform raw user input from
    client-side
    """

    global saved_bc_scaler, saved_bc_Y_le
    saved_bc_scaler = load_model('./modelling/saved/misc/bc_scaler.pkl')
    saved_bc_Y_le = load_model('./modelling/saved/misc/bc_Y_le.pkl')

load_models()
load_preprocessors()



# upon loading of client side fetch the model names
@app.route('/model-names', methods=['GET'])
def retrieve_model_names():
    """
    flask app will run at http://127.0.0.1:5000 if /
    in url succeeds another string <some string> then
    app will run at http://127.0.0.1:5000/<some string>

    returns json object of all model names loaded upon
    start of server and subsequent request of client to
    this view function
    """

    data = {
        'model_names': model_names
    }

    # return data at once since no error will most likely
    # occur on mere loading of the model
    return jsonify(data)

@app.route('/predict', methods=['POST'])
def predict():
    # extract raw data from client
    raw_data = request.json
    print(raw_data)

    # encoding and preprocessing
    radius_mean = float(raw_data['radius-mean'])
    texture_mean = float(raw_data['texture-mean'])
    perimeter_mean = float(raw_data['perimeter-mean'])
    area_mean = float(raw_data['area-mean'])
    smoothness_mean = float(raw_data['smoothness-mean'])

    # once x features are collected normalize the array on the 
    # saved scaler
    X = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]
    X_normed = saved_bc_scaler.transform(X)
    
    # predictor
    Y_preds = models[0].predict(X_normed)
    decoded_sparse_Y_preds = saved_bc_Y_le.inverse_transform(Y_preds)
    translated_labels = translate_labels(decoded_sparse_Y_preds, translations={'M': 'Malignant', 'B': 'Benign'})

    return jsonify({'diagnosis': translated_labels})

@app.route('/send-data', methods=['POST'])
def test_predict_a():
    # extract raw data from client
    raw_data = request.form
    raw_files = request.files
    print(raw_data)
    print(raw_files)

    first_name = raw_data['first_name']
    last_name = raw_data['last_name']
    email_address = raw_data['email_address']
    country_code = raw_data['country_code']
    mobile_num = raw_data['mobile_num']
    message = raw_data['message']
    model_name = raw_data['model_name']
    prompt = raw_data['prompt']
    seq_len = int(raw_data['seq_len'])
    temperature = float(raw_data['temperature'])
    image = raw_files['image']

    # preprocessing/encoding image stream into a matrix
    encoded_img = encode_image(image.stream)
    rescaled_img = standardize_image(encoded_img)
    print(rescaled_img.max())
    print(rescaled_img.shape)

    # predictor
    
    # reshape the image since the model takes in an (m, 256, 256, 3)
    # input, or in this case a single (1, 256, 256, 3) input
    img_shape = rescaled_img.shape
    reshaped_img = np.reshape(rescaled_img, newshape=(1, img_shape[0], img_shape[1], img_shape[2]))
    
    # predictor
    logits = models[0].predict(reshaped_img)

    # decoding stage
    Y_preds = activate_logits(logits)
    Y_preds = decode_one_hot(Y_preds)
    final_preds = re_encode_sparse_labels(Y_preds, new_labels=['Amoeba', 'Euglena', 'Hydra', 'Paramecium', 'Rod_bacteria', 'Spherical_bacteria', 'Spiral_bacteria', 'Yeast'])
    print(final_preds)
    
    return jsonify({'prediction': final_preds.tolist()})