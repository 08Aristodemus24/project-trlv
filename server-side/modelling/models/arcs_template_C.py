import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Activation, Embedding
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy as cce_metric
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import keras_tuner as kt



def init_embedding_layer(vocab_len, emb_matrix):
    emb_dim = emb_matrix.shape[1]
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


def load_lstm_model(input_shape, vocab_len, emb_matrix):
    """
    Define architecture of LSTM model then return for later training

    Takes in also the embedding layer with weights/coefficients set
    to the pre-trained GloVe embeddings
    """
    print(f'input shape: {input_shape}')

    seqs_padded = Input(shape=(input_shape, ), dtype='int64')
    embeddings = init_embedding_layer(vocab_len, emb_matrix)(seqs_padded)
    A1 = Bidirectional(LSTM(units=16, return_sequences=True))(embeddings)
    D1 = Dropout(0.5)(A1)
    A2 = Bidirectional(LSTM(units=16, return_sequences=False))(D1)
    D2 = Dropout(0.5)(A2)
    Z3 = Dense(units=3)(D2)
    A3 = Activation('softmax')(Z3)

    model = Model(inputs=seqs_padded, outputs=A3, name='hate-speech-lstm')

    return model

def compile_model(raw_model, X, learning_rate):
    raw_model.compile(
        loss=cce_loss(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[cce_metric(), CategoricalAccuracy()]
    )
    raw_model(X)
    raw_model.summary()

    return raw_model

def train_model(compiled_model, X, Y, epochs, batch_size):
    # define checkpoint and early stopping callback to save
    # best weights at each epoch and to stop if there is no improvement
    # of validation loss for 10 consecutive epochs
    weights_path = f"./saved/models/test_{compiled_model.name}" + "_{epoch:02d}_{val_categorical_accuracy:.2f}.h5"
    checkpoint = ModelCheckpoint(
        weights_path,
        monitor='val_categorical_accuracy',
        verbose=1,
        save_best_only=True,

        # used if model subclass is used
        # save_weights_only=True,
        mode='max')
    stopper = EarlyStopping(monitor='val_categorical_accuracy', patience=10)
    callbacks = [checkpoint, stopper]

    # begin training test model
    history = compiled_model.fit(X, Y,
        epochs=epochs,
        batch_size=batch_size, 
        callbacks=callbacks,
        validation_split=0.3,
        verbose=2,)
    
    return history

class HOSClassifierHyperModel(kt.HyperModel):
    def __init__(self, vocab_len, emb_matrix, name=None, tunable=True):
        """
        this will be the model builder that will be used
        for the tuner to find the optimal hyper parameters
        of the baseline architecture above

        the architecture of this model will of course be delimited
        to or example number of layers since deep language models
        are computationally expensive and so not all hyper params
        like number of layers will be tuned

        all in all this will still have a similar architecture
        to that of the baseline
        """
        super().__init__(name, tunable)
        self.vocab_len = vocab_len
        self.emb_matrix = emb_matrix

    def build(self, hp):
        # the drop probability values, instead of keep probability
        hp_dropout = hp.Choice('dropout', values=[0.1, 0.2, 0.3, 0.4, 0.5])

        # learning rate alpha
        hp_learning_rate = hp.Choice('learning_rate', values=[1.2, 0.03, 0.01, 0.0075, 0.003, 0.001])

        # regularization value lambda
        hp_lambda = hp.Choice('lambda', values=[1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.25, 0.125, 0.01])

        hp_num_rnn_units = hp.Choice('n_rnn_units', values=[16, 32, 64, 128, 256])

        # build architecture
        model = Sequential(name='hate-offensive-speech-classifier')

        model.add(init_embedding_layer(self.vocab_len, self.emb_matrix))
        model.add(Bidirectional(LSTM(units=hp_num_rnn_units, return_sequences=True)))
        model.add(Dropout(hp_dropout))
        model.add(Bidirectional(LSTM(units=hp_num_rnn_units, return_sequences=False)))
        model.add(Dropout(hp_dropout))
        model.add(Dense(units=3, kernel_regularizer=L2(hp_lambda)))
        model.add(Activation(activation=tf.nn.softmax))

        # define loss, optimizer, and metrics
        loss = cce_loss()
        opt = Adam(learning_rate=hp_learning_rate)
        metrics = [cce_metric(), CategoricalAccuracy()]
        model.compile(
            loss=loss,
            optimizer=opt,
            metrics=metrics
        )

        return model
    
def load_tuner(hyper_model, metric='val_accuracy', objective='max', max_epochs=10, factor=3, save_path: str='./saved/tuned_models'):
    
    obj = kt.Objective(metric, objective)

    tuner = kt.Hyperband(
        hyper_model, 
        objective=obj, 
        max_epochs=max_epochs,
        factor=factor,
        directory=save_path,
        project_name='tuned_model'
    )

    return tuner

def train_tuner(tuner, X, Y, epochs=10, batch_size=512):
    # define checkpoint and early stopping callback to save
    # best weights at each epoch and to stop if there is no 
    # improvement of validation loss for 3 consecutive epochs
    # since we are only using a tuner
    stopper = EarlyStopping(monitor='val_categorical_accuracy', patience=3)
    callbacks = [stopper]
    
    # fit model to data
    tuner.search(
        X, Y, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_split=0.3, 
        callbacks=callbacks
    )

    best_params = tuner.get_best_hyperparameters()[0]

    return best_params
