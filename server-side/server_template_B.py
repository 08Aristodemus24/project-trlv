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

# import and load model architectures as well as decoder
from modelling.models.arcs_template_B import GenPhiloTextA, generate
from modelling.utilities.preprocessors import (
    decode_id_sequences, 
    map_value_to_index, 
    lower_words, 
    clean_tweets, 
    remove_contractions, 
    rem_non_alpha_num, 
    rem_numeric, 
    rem_stop_words, 
    stem_corpus_words, 
    lemmatize_corpus_words, 
    strip_final_corpus,
    pad_token_sequences,
    decode_one_hot, 
    translate_labels,
)
    
from modelling.utilities.loaders import (
    load_lookup_table, 
    load_hyper_params, 
    load_tokenizer,
    load_model as model_loader)

import tensorflow as tf

# # configure location of build file and the static html template file
app = Flask(__name__, template_folder='static')

# since simple html from url http://127.0.0.1:5000 requests to
# api endpoint at http://127.0.0.1:5000/ we must set the allowed
# origins or web apps with specific urls like http://127.0.0.1:5000
# to be included otherwise it will be blocked by CORS policy
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5000", "https://gen-philo-text.vercel.app", "https://gen-philo-text.onrender.com"])

# global variables
vocab = None
char_to_idx = None
idx_to_char = None
hyper_params = None
model = None
tuned_model = None
tokenizer = None

# functions to load miscellaneous variables, hyperparameters, and the model itself
def load_misc():
    """
    loads miscellaneous variables to be used by the model
    """
    global vocab, char_to_idx, idx_to_char, hyper_params
    
    vocab = load_lookup_table('./modelling/final/misc/char_to_idx')
    char_to_idx, idx_to_char = map_value_to_index(vocab)
    hyper_params = load_hyper_params('./modelling/final/misc/hyper_params.json')

def load_preprocessors():
    """
    prepares and loads the saved encoders, normalizers,
    and tokenizers of the dataset to later transform 
    raw user input from client-side
    """

    global saved_tokenizer, saved_hos_Y_le
    saved_tokenizer = load_tokenizer('./saved/misc/tokenizer.json')
    saved_hos_Y_le = model_loader('./saved/misc/hos_Y_le.pkl')


def load_model_weights():
    """
    prepares and loads sample input and custom model in
    order to use trained weights/parameters/coefficients
    """

    # declare sample input in order ot 
    # use load_weights() method
    sample_input = tf.random.uniform(shape=(1, hyper_params['T_x']), minval=0, maxval=hyper_params['n_unique'] - 1, dtype=tf.int32)
    sample_h = tf.zeros(shape=(1, hyper_params['n_a']))
    sample_c = tf.zeros(shape=(1, hyper_params['n_a']))

    # recreate model architecture
    global model
    model = GenPhiloTextA(
        emb_dim=hyper_params['emb_dim'],
        n_a=hyper_params['n_a'],
        n_unique=hyper_params['n_unique'],
        dense_layers_dims=hyper_params['dense_layers_dims'] + [hyper_params['n_unique']],
        lambda_=hyper_params['lambda_'],
        drop_prob=hyper_params['drop_prob'],
        normalize=hyper_params['normalize'])
    
    # call model on sample input before loading weights
    model(sample_input)

    # load weights
    model.load_weights('./modelling/final/weights/notes_gen_philo_text_a_100_3.0299.h5')

def load_model():
    """
    prepares and loads a saved .h5 model instead iof its
    weights in order to quickly load both architecture
    and its optimized coefficeints/weights/parameters
    """

    global tuned_model
    tuned_model = tf.keras.models.load_model('./modelling/saved/models/tuned_hate-speech-lstm.h5')


load_misc()
load_preprocessors()
load_model_weights()
load_model()



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # extract raw data from client
    raw_data = request.json
    print(raw_data)

    # encoder to preprocess prompt input from client
    prompt = remove_contractions(raw_data['prompt'])
    prompt = rem_non_alpha_num(prompt)
    prompt = [prompt]

    temperature = float(raw_data['temperature'])

    T_x = int(raw_data['sequence_length'])
    

    # predictor
    pred_ids = generate(model, prompts=prompt, char_to_idx=char_to_idx, T_x=T_x, temperature=temperature)

    # decoder
    decoded_ids = decode_id_sequences(pred_ids, idx_to_char=idx_to_char)

    return jsonify({'message': decoded_ids})

@app.route('/tuned-predict', methods=['POST'])
def tuned_predict():
    # extract raw data from client
    raw_data = request.json
    tweet = raw_data['tweet']

    # preprocess tweet using preprocessors used in training
    # which is the encoder part of this pipeline
    corpus = lower_words(tweet)
    corpus = clean_tweets(corpus)
    corpus = remove_contractions(corpus)
    corpus = rem_non_alpha_num(corpus)
    corpus = rem_numeric(corpus)
    corpus = rem_stop_words(corpus)
    corpus = stem_corpus_words(corpus)
    corpus = lemmatize_corpus_words(corpus)
    corpus = strip_final_corpus(corpus)

    # this is where we will need to access the Tokenizer we
    # trained on the training data by saving it prior to
    # usage on server
    seqs = tokenizer.texts_to_sequences([corpus])
    padded_seqs = pad_token_sequences(seqs)

    # predictor
    Y_preds = tuned_model.predict(padded_seqs)

    # decoder
    sparse_Y_preds = decode_one_hot(Y_preds)
    decoded_sparse_Y_preds = saved_hos_Y_le.inverse_transform(sparse_Y_preds)
    translated_labels = translate_labels(decoded_sparse_Y_preds)

    jsonify({'prediction': translated_labels})

@app.errorhandler(404)
def page_not_found(error):
    print(error)
    return 'This page does not exist', 404



