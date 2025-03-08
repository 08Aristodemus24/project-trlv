from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ff. imports are for getting secret values from .env file
from pathlib import Path
import os

from modelling.utilities.loaders import load_model
from modelling.utilities.preprocessors import (
    lower_words,
    remove_contractions,
    rem_non_alpha_num,
    rem_numeric,
    rem_stop_words,
    stem_corpus_words,
    lemmatize_corpus_words,
    strip_final_corpus,
    translate_labels
)

# configure location of build file and the static html template file
app = Flask(__name__, template_folder='static')

# since simple html from url http://127.0.0.1:5000 requests to
# api endpoint at http://127.0.0.1:5000/ we must set the allowed
# origins or web apps with specific urls like http://127.0.0.1:5000
# to be included otherwise it will be blocked by CORS policy
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5000",])

models = {}
saved_ddr_tfidf_vec = None
saved_ddr_le = None

def load_models():
    """
    prepares and loads sample input and custom model in
    order to use trained weights/parameters/coefficients
    """

    # recreate model architecture
    saved_lgbm_clf = load_model('./modelling/saved/models/lgbm_clf.pkl')
    saved_xgb_clf = load_model('./modelling/saved/models/xgb_clf.pkl')
    saved_ada_clf = load_model('./modelling/saved/models/ada_clf.pkl')
    models[type(saved_lgbm_clf).__name__] = saved_lgbm_clf
    models[type(saved_xgb_clf).__name__] = saved_xgb_clf
    models[type(saved_ada_clf).__name__] = saved_ada_clf

def load_preprocessors():
    """
    prepares and loads the saved encoders, normalizers of
    the dataset to later transform raw user input from
    client-side
    """

    global saved_ddr_tfidf_vec, saved_ddr_le
    saved_ddr_tfidf_vec = load_model('./modelling/saved/misc/ddr_tfidf_vec.pkl')
    saved_ddr_le = load_model('./modelling/saved/misc/ddr_le.pkl')

load_models()
load_preprocessors()

@app.route('/')
def index():
    return render_template('index.html')

@app.errorhandler(404)
def page_not_found(error):
    print(error)
    return 'This page does not exist', 404

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
        'model_names': list(models.keys())
    }

    # return data at once since no error will most likely
    # occur on mere loading of the model
    return jsonify(data)

@app.route('/predict', methods=['POST'])
def predict():
    # extract raw data from client
    raw_data = request.form
    print(raw_data)

    # encoding and preprocessing
    message = raw_data['message']
    message = lower_words(message)
    message = remove_contractions(message)
    message = rem_non_alpha_num(message)
    message = rem_numeric(message)
    message = rem_stop_words(message)
    message = stem_corpus_words(message)
    message = lemmatize_corpus_words(message)
    message = [strip_final_corpus(message)]

    model_name = raw_data['model_name']
    print(model_name)
    model = models[model_name]

    # once x features are collected normalize the array on the 
    # saved scaler
    X_vec = saved_ddr_tfidf_vec.transform(message)
    print(X_vec)
    
    # predictor
    Y_preds = model.predict(X_vec)
    print(Y_preds)
    decoded_sparse_Y_preds = saved_ddr_le.inverse_transform(Y_preds)
    print(decoded_sparse_Y_preds)
    translated_labels = translate_labels(decoded_sparse_Y_preds, translations={'DPR': 'Depressive', 'NDP': 'Non-Depressive'})
    print(translated_labels)

    return jsonify({'sentiment': translated_labels.tolist()})