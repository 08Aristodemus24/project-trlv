# %% [markdown]
# ## Ensemble model training
# - this notebook will serve to extract the best hyper parameters

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler

import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy as bce_loss
from tensorflow.keras.metrics import BinaryAccuracy, BinaryCrossentropy as bce_metric
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform, RandomNormal, RandomUniform, HeNormal, HeUniform
from tensorflow.keras.optimizers import Adadelta, Adafactor, Adagrad, Adam, AdamW, Adamax, Ftrl, Nadam, RMSprop, SGD 

from utilities.data_preprocessor import preprocess

%load_ext autoreload
%autoreload 2

# %% [markdown]
# ## Define architecture and hyper parameters to use

# %%
def model_builder(hp):
    """
    hp - hyperparameter
    """

    model = Sequential()

    hp_kernel_initializer = hp.Choice('initializer', values=['GlorotNormal', 'GlorotUniform', 'RandomNormal', 'RandomUniform', 'HeNormal', 'HeUniform'])
    initializers = {
        'GlorotNormal': GlorotNormal(),
        'GlorotUniform': GlorotUniform(),
        'RandomNormal': RandomNormal(mean=0.0, stddev=1.0),
        'RandomUniform': RandomUniform(minval=-0.05, maxval=0.05),
        'HeNormal': HeNormal(),
        'HeUniform': HeUniform()
    }

    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])

    # the drop probability values, instead of keep probability
    hp_dropout = hp.Choice('dropout', values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # learning rate alpha
    hp_learning_rate = hp.Choice('learning_rate', values=[1.2, 0.03, 0.01, 0.0075, 0.003, 0.001,])

    # regularization value lambda
    hp_lambda = hp.Choice('lambda', values=[10.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.25, 0.125, 0.01,])
    # hp_dropout = hp.Choice('dropout', value=[0.8, 0.85, 0.7, 0.6])

    hp_optimizer = hp.Choice('optimizer', values=['Adadelta', 'Adafactor', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD'])
    optimizers = {
        'Adadelta': Adadelta(learning_rate=hp_learning_rate),
        'Adafactor': Adafactor(learning_rate=hp_learning_rate),
        'Adagrad': Adagrad(learning_rate=hp_learning_rate),
        'Adam': Adam(learning_rate=hp_learning_rate),
        'AdamW': AdamW(learning_rate=hp_learning_rate),
        'Adamax': Adamax(learning_rate=hp_learning_rate), 
        'Ftrl': Ftrl(learning_rate=hp_learning_rate),
        'Nadam': Nadam(learning_rate=hp_learning_rate),
        'RMSprop': RMSprop(learning_rate=hp_learning_rate),
        'SGD': SGD(learning_rate=hp_learning_rate)
    }

    # number of hidden layers
    for index, l in enumerate(range(hp.Int('layer_num', min_value=1, max_value=80))):
        # number of nodes per layer
        model.add(Dense(
            units=hp.Int(f'layer_{index + 1}', min_value=1, max_value=1000, step=100), 
            activation=hp_activation, 
            kernel_initializer=initializers[hp_kernel_initializer],
            kernel_regularizer=L2(hp_lambda)))
        
        model.add(Dropout(hp_dropout))

    model.add(Dense(units=1, activation='linear', kernel_regularizer=L2(hp_lambda)))
    
    model.compile(
        optimizer=optimizers[hp_optimizer],
        loss=bce_loss(from_logits=True),
        metrics=[bce_metric(), BinaryAccuracy(threshold=0.5)]
    )

    return model

# %%
# define tuner
tuner = kt.Hyperband(
    model_builder, 
    objective=kt.Objective('val_binary_crossentropy', 'min'), 
    max_epochs=100,
    factor=3,
    directory='tuned_models',
    project_name='model'
)

# if cross validation loss does not improve after 10 
# consecutive epochs we stop training our modelearly
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# %% [markdown]
# ## load data
# 

# %%
# load data with the selected features to use
df = pd.read_csv('./data.csv')
X, Y = preprocess(df)


# %%
X

# %% [markdown]
# ## ensemble train the multiple models

# %%
# fit model to data
tuner.search(
    X, Y, 
    epochs=50, 
    validation_split=0.3, 
    callbacks=[stop_early]
)

# %% [markdown]
# ## extract the hyper parameters of the best model that trained 

# %%
hidden_layer_num = tuner.get_best_hyperparameters()[0].get('layer_num')

# %%
hp_names = [f"layer_{l + 1}" for l in range(hidden_layer_num)] + ['activation', 'learning_rate', 'lambda', 'optimizer', 'dropout', 'initializer', 'layer_num']
best_hyper_params = {}
for hp in hp_names:
    best_hyper_param = tuner.get_best_hyperparameters()[0].get(hp)
    print(f'{hp}: {best_hyper_param}')

    if hp not in best_hyper_params:
        best_hyper_params[hp] = best_hyper_param


best_hps = tuner.get_best_hyperparameters()[0]
best_hps


# %% [markdown]
# ## save best hyper parameter values to file

# %%
import json

best_hyper_params

# %%
with open('./results/best_hyper_params.json', 'w') as out_file:
    json.dump(best_hyper_params, out_file)

# %%



