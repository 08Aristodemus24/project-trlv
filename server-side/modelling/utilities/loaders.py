import csv
import numpy as np
import tqdm
import pickle
import json
import os
import pandas as pd

from pathlib import Path
from splitfolders import ratio

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import tokenizer_from_json


def device_exists():
    """
    returns true if gpu device exists
    """

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        return False
    return True

# for NLP data
def load_corpus(path: str):
    """
    reads a text file and returns the text
    """

    with open(path, 'r', encoding='utf-8') as file:
        corpus = file.read()

    return corpus

def get_chars(corpus: str):
    """
    returns a list of all unique characters found
    in given corpus
    """

    chars = sorted(list(set(corpus)))

    return chars

def load_lookup_array(path: str):
    """
    reads a text file containing a list of all unique values
    and returns this
    """

    with open(path, 'rb') as file:
        char_to_idx = pickle.load(file)
        file.close()

    return char_to_idx

def save_lookup_array(path: str, uniques: list):
    """
    saves and writes all the unique list of values to a
    a file for later loading by load_lookup_array()
    """

    with open(path, 'wb') as file:
        pickle.dump(uniques, file)
        file.close()

def save_meta_data(path: str, meta_data: dict):
    """
    saves dictionary of meta data such as hyper 
    parameters to a .json file
    """

    with open(path, 'w') as file:
        json.dump(meta_data, file)
        file.close()

def load_meta_data(path: str):
    """
    loads the saved dictionary of meta data such as
    hyper parameters from the created .json file
    """

    with open(path, 'r') as file:
        meta_data = json.load(file)
        file.close()

    return meta_data

def save_model(model, path: str):
    """
    saves partcularly an sklearn model in a .pkl file
    for later testing
    """

    with open(path, 'wb') as file:
        pickle.dump(model, file)
        file.close()

def load_model(path: str):
    """
    loads the sklearn model, scaler, or encoder stored
    in a .pkl file for later testing and deployment
    """

    with open(path, 'rb') as file:
        model = pickle.load(file)
        file.close()

    return model

def save_tokenizer(path: str, tokenizer):
    """
    saves the tokenizer fitted on the NLP training data 
    """
    tokenizer_json = tokenizer.to_json()
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(tokenizer_json, file, ensure_ascii=False)
        file.close()

def load_tokenizer(path: str):
    """
    The data can be loaded using tokenizer_from_json function 
    from keras_preprocessing.text
    """
    with open(path, 'r') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    return tokenizer

def construct_embedding_dict(emb_path):
    """
    returns an embedding dictionary populated with all the unique 
    values and their respective vector representations from the 
    file of the pretrained embeddings 

    creates the embedding dictionary from the text file containing
    the pre-trained embeddings which are in the format below:

    the 0.1234 0.423 -0.1324 ... -0.231
    hello 0.1234 0.423 -0.1324 ... -0.231
    nice 0.1234 0.423 -0.1324 ... -0.231
    """

    with open(emb_path, encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=' ', quoting=csv.QUOTE_NONE)

        # map(float, line[1:]) takes the read line which is 
        # an array of the flaot values excluding the word and
        # passes each of its elements to the callback we 
        # provided which was `float`
        embedding_dict = {line[0]: np.array(list(map(float, line[1:]))) for line in reader}

    # gets the first item of the created dictionary
    first_key = next(iter(embedding_dict))

    # access the value or vector of the given the 
    # key and get its length
    emb_dim = embedding_dict[first_key].shape[0]

    return embedding_dict, emb_dim

def construct_embedding_matrix(val_to_index, embedding_dict, emb_dim):
    """
    Constructs the embedding matrix upon finishing the phase of 
    constructing the embedding dictionary. So that reading the
    embeddings is only done once to increase time efficiency
    """

    # oov words (out of vacabulary words) will be mapped to 0 vectors
    # this is why we have a plus one always to the number of our words in 
    # our embedding matrix since that is reserved for an unknown or OOV word
    vocab_len = len(val_to_index) + 1

    # initialize it to 0
    embedding_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in tqdm.tqdm(val_to_index.items()):
        # skip if, if index is already equal to the number of
        # words in our vocab. A break statement if you will
        if index < vocab_len:
            # if word does not exist in the pretrained word embedding itself
            # then return an empty array
            vector = embedding_dict.get(word, [])

            # if in cases vect is indeed otherwise an empty array due 
            # to the word existing in the pretrained word embeddings
            # then place it in our embedding matrix. Otherwise its index
            # where a word does not exist will stay a row of zeros
            if len(vector) > 0:
                embedding_matrix[index] = vector[:emb_dim]

    return embedding_matrix

def get_cat_cols(df):
    """
    returns all categorical columns/features names
    as a list
    """

    cols = df.columns

    num_cols = df._get_numeric_data().columns.to_list()

    # get complement of set of columns and numerical columns
    cat_cols = list(set(cols) - set(num_cols))
    
    return cat_cols

def get_top_models(models_train, models_cross, pool_size: int=10, model_type: str="regressor"):
    """
    takes in the dataframes returned by either LazyClassifier or LazyPredict
    e.g. clf = LazyRegressor(
        verbose=0, 
        ignore_warnings=True, 
        custom_metric=None, 
        regressors=[LinearRegression, Ridge, Lasso, DecisionTreeRegressor, RandomForestRegressor, XGBRegressor, SVR])
    models_train, predictions_train = clf.fit(ch_X_trains, ch_X_trains, ch_Y_trains, ch_Y_trains)
    models_cross, predictions_cross = clf.fit(ch_X_trains, ch_X_cross, ch_Y_trains, ch_Y_cross)

    args:
        models_train - 
        models_cross - 
        pool_size - number of rows to take into consideration when merging the
        dataframes of model train and cross validation metric values
    """

    # rename columns for each dataframe to avoid duplication during merge
    for col in models_train.columns:
        models_train.rename(columns={f"{col}": f"Train {col}"}, inplace=True)
        models_cross.rename(columns={f"{col}": f"Cross {col}"}, inplace=True)

    # merge both first pool_size rows of training and cross 
    # validation model dataframes
    models_train = models_train[:pool_size].reset_index()
    models_cross = models_cross[:pool_size].reset_index()
    
    # merge model dataframes on 'Model' column
    top_models = models_train.merge(models_cross, how='inner', left_on='Model', right_on='Model')
    top_models.sort_values(by="Cross Adjusted R-Squared" if model_type == "regressor" else "Cross F1 Score", inplace=True, ascending=False)

    return top_models

def create_image_set(root_dir: str, img_dims: tuple=(256, 256)):
    temp = sorted(os.listdir(root_dir))

    # creates new copies of the subdirectories of train, cross, and
    # testing folders under each class/label subdirectory e.g. 
    # Amoeba will have now train, cross, and testing folders in it
    sub_dirs = Path(root_dir)
    output_dir = f'{root_dir[:-1]}_Split'
    ratio(sub_dirs, output=output_dir, seed=0, ratio=(0.7, 0.15, 0.15), group_prefix=None)

    # augments the unbalanced image data we currently have
    # by rotating, flipping, distorting images to produce
    # more of a balanced image set
    gen = ImageDataGenerator(
        # instead of our rgb values being between 0 and 255 doing 
        # this rescales the rgb values between 0 and 1
        rescale=1.0 / 255,

        # degree range for random rotations.
        rotation_range=10,

        # Randomly flip inputs horizontally
        horizontal_flip=True,

        vertical_flip=True,

        # values lie between 0 being dark and 1 being bright
        brightness_range=[0.3, 0.8]
    )

    # gen.flow_from_directory actually returns a generator object
    # which recall we can use the next() with to get the next element
    train_gen = gen.flow_from_directory(
        # this arg should contain one subdirectory per class. Any PNG, JPG, 
        # BMP, PPM or TIF images inside each of the subdirectories directory 
        # tree will be included in the generator
        f'{output_dir}/train',
        target_size=img_dims,

        # means the labels/targets produced by generator will be
        # a one encoding of each of our different classes e.g.
        # amoeba will have [1, 0, 0, 0, 0, 0, 0, 0]
        class_mode='categorical',
        subset='training',
        batch_size=128
    )

    cross_gen = gen.flow_from_directory(
        f'{output_dir}/val',
        target_size=img_dims,
        class_mode='categorical',
        shuffle=False
    )

    test_gen = gen.flow_from_directory(
        f'{output_dir}/test',
        target_size=img_dims,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, cross_gen, test_gen

def create_metrics_df(train_metric_values, 
                      val_metric_values, 
                      test_metric_values, 
                      metrics=['accuracy', 'precision', 'recall', 'f1-score']):
    """
    creates a metrics dataframe
    """

    metrics_dict = {
        'data_split': ['training', 'validation', 'testing']
    }

    for index, metric in enumerate(metrics):
        metrics_dict[metric] = [
            train_metric_values[index], 
            val_metric_values[index],
            test_metric_values[index]
        ]

    metrics_df = pd.DataFrame(metrics_dict)

    return metrics_df

def create_classified_df(train_conf_matrix, val_conf_matrix, test_conf_matrix, train_labels, val_labels, test_labels):
    """
    creates a dataframe that represents all classified and 
    misclassified values
    """

    num_right_cm_train = train_conf_matrix.trace()
    num_right_cm_val = val_conf_matrix.trace()
    num_right_cm_test = test_conf_matrix.trace()

    num_wrong_cm_train = train_labels.shape[0] - num_right_cm_train
    num_wrong_cm_val = val_labels.shape[0] - num_right_cm_val
    num_wrong_cm_test = test_labels.shape[0] - num_right_cm_test

    classified_df = pd.DataFrame({
        'data_split': ['training', 'validation', 'testing'],
        'classified': [num_right_cm_train, num_right_cm_val, num_right_cm_test], 
        'misclassified': [num_wrong_cm_train, num_wrong_cm_val, num_wrong_cm_test]}, 
        index=["training set", "validation set", "testing set"])
    
    return classified_df
