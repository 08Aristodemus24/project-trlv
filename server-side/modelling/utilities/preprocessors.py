import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from PIL import Image

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pandas as pd
import numpy as np
import ast
import tensorflow as tf

# download stopwords and wordnet if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')

def encode_features(X):
    """
    encodes the categorical features of a dataset into numerical values
    given the desired feature to encode and the input X to transform

    if shape of input is a one dimensional array and not a typical
    matrix reshape it to an m x 1 matrix instead by expanding its 
    dimensions. Usually this will be a the target column of 1 
    dimension. Otherwise use the ordinal encoder which is suitable for
    matrices like the set of independent variables of an X input

    used during training, validation, and testing/deployment (since
    encoder is saved and later used)
    """

    enc = LabelEncoder() if len(X.shape) < 2 else OrdinalEncoder(dtype=np.int64)
    enc_feats = enc.fit_transform(X)
    return enc_feats, enc

def normalize_train_cross(X_trains, X_cross, scaler='min_max'):
    """
    normalizes training and cross validation datasets using either
    a standard z-distribution or min max scaler

    args:
        X_trains - 
        X_cross - 
        scaler - scaler to use which can either be 'min_max' or 'standard'

    used during training, validation, and testing/deployment (since
    scaler is saved and later used)
    """

    temp = MinMaxScaler() if scaler is 'min_max' else StandardScaler()
    X_trains_normed = temp.fit_transform(X_trains)
    X_cross_normed = temp.transform(X_cross)

    return X_trains_normed, X_cross_normed, temp


def map_value_to_index(unique_tokens: list):
    """
    returns a lookup table mapping each unique value to an integer. 
    This is akin to generating a word to index dictionary where each
    unique word based on their freqeuncy will be mapped from indeces
    1 to |V|.

    args:
        unique_tokens - an array consisting of all unique tokens

    returns:
        a lookup layer for each unique character/token, including
        the [UNK] token's corresponding id

    used during training, validation, and testing/deployment
    """
    char_to_idx = tf.keras.layers.StringLookup(vocabulary=unique_tokens, mask_token=None)
    idx_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_idx.get_vocabulary(), invert=True, mask_token=None)

    return char_to_idx, idx_to_char

def lower_words(corpus: str):
    """
    lowers all chars in corpus

    used during training, validation, and testing/deployment
    """
    print(corpus)

    return corpus.lower()

def remove_contractions(text_string: str):
    """
    removes contractions and replace them e.g. don't becomes
    do not and so on

    used during training, validation, and testing/deployment
    """

    text_string = re.sub(r"don't", "do not ", text_string)
    text_string = re.sub(r"didn't", "did not ", text_string)
    text_string = re.sub(r"aren't", "are not ", text_string)
    text_string = re.sub(r"weren't", "were not", text_string)
    text_string = re.sub(r"isn't", "is not ", text_string)
    text_string = re.sub(r"can't", "cannot ", text_string)
    text_string = re.sub(r"doesn't", "does not ", text_string)
    text_string = re.sub(r"shouldn't", "should not ", text_string)
    text_string = re.sub(r"couldn't", "could not ", text_string)
    text_string = re.sub(r"mustn't", "must not ", text_string)
    text_string = re.sub(r"wouldn't", "would not ", text_string)

    text_string = re.sub(r"what's", "what is ", text_string)
    text_string = re.sub(r"that's", "that is ", text_string)
    text_string = re.sub(r"he's", "he is ", text_string)
    text_string = re.sub(r"she's", "she is ", text_string)
    text_string = re.sub(r"it's", "it is ", text_string)
    text_string = re.sub(r"that's", "that is ", text_string)

    text_string = re.sub(r"could've", "could have ", text_string)
    text_string = re.sub(r"would've", "would have ", text_string)
    text_string = re.sub(r"should've", "should have ", text_string)
    text_string = re.sub(r"must've", "must have ", text_string)
    text_string = re.sub(r"i've", "i have ", text_string)
    text_string = re.sub(r"we've", "we have ", text_string)

    text_string = re.sub(r"you're", "you are ", text_string)
    text_string = re.sub(r"they're", "they are ", text_string)
    text_string = re.sub(r"we're", "we are ", text_string)

    text_string = re.sub(r"you'd", "you would ", text_string)
    text_string = re.sub(r"they'd", "they would ", text_string)
    text_string = re.sub(r"she'd", "she would ", text_string)
    text_string = re.sub(r"he'd", "he would ", text_string)
    text_string = re.sub(r"it'd", "it would ", text_string)
    text_string = re.sub(r"we'd", "we would ", text_string)

    text_string = re.sub(r"you'll", "you will ", text_string)
    text_string = re.sub(r"they'll", "they will ", text_string)
    text_string = re.sub(r"she'll", "she will ", text_string)
    text_string = re.sub(r"he'll", "he will ", text_string)
    text_string = re.sub(r"it'll", "it will ", text_string)
    text_string = re.sub(r"we'll", "we will ", text_string)

    text_string = re.sub(r"\n't", " not ", text_string) #
    text_string = re.sub(r"\'s", " ", text_string) 
    text_string = re.sub(r"\'ve", " have ", text_string) #
    text_string = re.sub(r"\'re", " are ", text_string) #
    text_string = re.sub(r"\'d", " would ", text_string) #
    text_string = re.sub(r"\'ll", " will ", text_string) # 
    
    text_string = re.sub(r"i'm", "i am ", text_string)
    text_string = re.sub(r"%", " percent ", text_string)
    print(text_string)

    return text_string

def rem_non_alpha_num(corpus: str):
    """
    removes all non-alphanumeric values in the given corpus

    used during training, validation, and testing/deployment
    """
    print(corpus)
    return re.sub(r"[^0-9a-zA-ZñÑ.\"]+", ' ', corpus)

def rem_numeric(corpus: str):
    print(corpus)
    return re.sub(r"[0-9]+", ' ', corpus)

def capitalize(corpus: str):
    """
    capitalizes all individual words in the corpus

    used during training, validation, and testing/deployment
    """
    print(corpus)
    return corpus.title()

def filter_valid(corpus: str, to_exclude: list=
        ['Crsp', 'Rpm', 'Mapsy', 'Cssgb', 'Chra', 
        'Mba', 'Es', 'Csswb', 'Cphr', 'Clssyb', 
        'Cssyb', 'Mdrt', 'Ceqp', 'Icyb']):
    
    """
    a function that filters only valid names and
    joins only the words that is valid in the profile
    name e.g. 'Christian Cachola Chrp Crsp'
    results only in 'Christian Cachola'

    used during training, validation, and testing/deployment
    """

    # filter and remove the words in the sequence
    # included in list of words that are invalid
    sequence = corpus.split()
    filt_sequence = list(filter(lambda word: word not in to_exclude, sequence))
    
    # join the filtered words
    temp = " ".join(filt_sequence)

    return temp

def partition_corpus(corpus: str):
    """
    splits a corpus like name, phrase, sentence, 
    paragraph, or corpus into individual strings

    used during training, validation, and testing/deployment
    """
    print(corpus)

    return corpus.split()

def rem_stop_words(corpus: str, other_exclusions: list=["#ff", "ff", "rt", "amp"]):
    """
    removes stop words of a given corpus

    used during training, validation, and testing/deployment
    """

    # get individual words of corpus
    words = corpus.split()

    # extract stop words and if provided with other exclusions
    # extend this to the list of stop words
    stop_words = stopwords.words('english')
    stop_words.extend(other_exclusions)

    # include only the words not in the list of stop words
    words = [word for word in words if not word in stop_words]

    # rejoin the individual words of the now removed stop words
    corpus = " ".join(words)
    print(corpus)

    return corpus

def stem_corpus_words(corpus: str):
    """
    stems individual words of a given corpus

    used during training, validation, and testing/deployment
    """
    # get individual words of corpus
    words = corpus.split()

    # stem each individual word in corpus
    snowball = SnowballStemmer("english", ignore_stopwords=True)
    words = [snowball.stem(word) for word in words]

    # rejoin the now lemmatized individual words
    corpus = " ".join(words)
    print(corpus)

    return corpus

def lemmatize_corpus_words(corpus: str):
    """
    lemmatizes individual words of a given corpus

    used during training, validation, and testing/deployment
    """

    # get individual words of corpus
    words = corpus.split()

    # lemmatize each individual word in corpus
    wordnet = WordNetLemmatizer()
    words = [wordnet.lemmatize(word) for word in words]

    # rejoin the now lemmatized individual words
    corpus = " ".join(words)
    print(corpus)

    return corpus

def clean_tweets(corpus):
    """
    Accepts a text string or corpus and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned

    used during training, validation, and testing/deployment
    """

    space_pattern = '\s+'

    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
    '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    mention_regex = '@[\w\-]+'

    text_string = re.sub(space_pattern, ' ', corpus)
    text_string = re.sub(giant_url_regex, '', text_string)
    text_string = re.sub(mention_regex, '', text_string)
    print(text_string)
    
    return text_string

def strip_final_corpus(corpus):
    """
    ideally used in final phase of preprocessing corpus/text string
    which strips the final corpus of all its white spaces

    used during training, validation, and testing/deployment
    """
    
    return corpus.strip()

def join_word_list(word_list):
    return " ".join(word_list)

def sentences2tok_sequences(n_unique, sents: pd.Series):
    """
    takes in a list or pandas series of sentences of str
    type and converts them to their respective unique indeces

    returns the sequence of now tokenized words as well as the
    word_to_index and index_to_word dictionaries/mappings

    used during training, validation, and testing/deployment
    (since tokenizer is saved)
    """

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=n_unique, 
        split=' '
    )
    tokenizer.fit_on_texts(sents)
    # the bug is here that's why there are wrong indeces

    seqs = tokenizer.texts_to_sequences(sents)
    word_to_index = tokenizer.word_index
    index_to_word = tokenizer.index_word

    return seqs, tokenizer, word_to_index, index_to_word

def pad_token_sequences(seqs, n_time_steps: int=50):
    """
    pads the tokenized sequences with zeroes so that the resulting
    dataset are tokenized sequences of the same length, the length
    defined by the user through the number of time steps argument
    or the maximum length each tokenized sequence should have

    args:
        n_time_steps - a user defined argument that can represents
        how long must the sequences of tokens/ids be, for most cases
        it would be the length of the longest sentence or the number
        of words in that longest sentence or just a shorter sequence
        for optimal performance without losing time efficiency

    used during training, validation, and testing/deployment
    """
    # post means place padding of 0's on the tail or ending of the sequence
    # and truncating removes the values of a sequence that is greater than the max length given
    seqs_padded = tf.keras.preprocessing.sequence.pad_sequences(
        seqs, 
        maxlen=n_time_steps, 
        padding='post', 
        truncating='post'
    )

    return seqs_padded

def string_list_to_list(column: pd.Series):
    """
    saving df to csv with a column that is of a list data type
    does not preserve its type and is converted instead to an
    str so convert first str to list or series "["a", "b", 
    "hello"]" to ["a", "b", "hello"]

    used only during training and validation
    """
    column = column.apply(lambda comment: ast.literal_eval(comment))

    return column

def flatten_series_of_lists(column: pd.Series):
    """this converts the series or column of a df
    of lists to a flattened version of it

    used only during training and validation
    """

    return pd.Series([item for sublist in column for item in sublist])

def sentences_to_avgs(sentences, word_to_vec_dict: dict):
    """
    Converts a series object of sentences (string) into a list of words (strings) then 
    extracts the GloVe representation of each word and averages its value into a single 
    vector encoding the meaning of the sentence. Return a 2D numpy array representing 
    all sentences vector representations
    
    args:
        sentences -- a series object of sentences
        word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    
    returns:
        avg -- average vector encoding information about each sentence, numpy-array of shape (m, d), where m is the
        number of sentences in the dataframe and d is the number of dimensions or the length of a word's vector rep

    used only during training and validation
    """
    
    # Initialize the average word vector, should have the same shape
    # as your word vectors. Use `np.zeros` and pass in the argument 
    # of any word's word 2 vec's shape. Get also a valid word contained
    # in the word_to_vec_map.
    any_word = list(word_to_vec_dict.keys())[0]
    m = sentences.shape[0]
    d = word_to_vec_dict[any_word].shape[0]
    avgs = np.zeros(shape=(m, d))

    # in each sentence split them into individual words then take the 
    # average of these occuring words in the word embedding dictionary
    for index, sentence in enumerate(sentences):
        # Initialize/reset count to 0
        count = 0

        # Step 1: Split sentence into list of lower case words 
        words = sentence.lower().split(' ')
        
        # Step 2: average the word vectors. You can loop over the words in the list "words".
        for word in words:
            # Check that word exists in word_to_vec_dict
            if word in word_to_vec_dict:
                
                # add the d-dim vector representation of the word 'word'
                # to the avg variable that will contain our summed 
                # vectors of a sentences words
                avgs[index] += word_to_vec_dict[word]
                
                # Increment count which represents how many words we have in our sentence
                count +=1
            
        if count > 0:
            # Get the average. But only if count > 0
            avgs[index] = avgs[index] / count
    
    return avgs

def init_sequences(corpus: str, char_to_idx: dict, T_x: int):
    """
    generates a input and target dataset by:

    1. partitioning corpus first into sequences of length T_x + 1
    2. shifting sequences by one character to the left to generate 
    output/target sequence the model needs to learn

    A sequence length of 0 will not be permitted and this
    function will raise an error should T_x be 0

    used only during training and validation
    """

    if T_x == 0:
        raise ValueError("You have entered an unpermitted value for the number of timesteps T_x. Sequence length T_x cannot be 0. Choose a value above 0.")

    # get total length of corpus
    total_len = len(corpus)

    # will contain our training examples serving as 
    # our x and y values
    in_seqs = []
    out_seqs = []

    # generate pairs of input and output sequences that 
    # will serve as our training examples x and y
    # loop through each character and every T_x char append it to the
    for i in range(0, total_len, T_x + 1):
        # slice corpus into input and output characters 
        # and convert each character into their respective 
        # indeces using char_to_idx mapping
        partition = [ch for ch in corpus[i: i + (T_x + 1)]]
        in_seq = partition[:-1]
        out_seq = partition[1:]

        # append input and output sequences
        in_seqs.append(in_seq)
        out_seqs.append(out_seq)

    if total_len % (T_x + 1):
        # calculate number of chars missing in last training example
        n_chars_missed = T_x - len(in_seqs[-1])

        # pad with zeroes to example with less than 100 chars
        in_seqs[-1] = in_seqs[-1] + (['[UNK]'] * n_chars_missed)
        out_seqs[-1] = out_seqs[-1] + (['[UNK]'] * n_chars_missed)

    return char_to_idx(in_seqs), char_to_idx(out_seqs)

def decode_id_sequences(pred_ids, idx_to_char):
    """
    decodes the sequence of id predictions by inference 
    model and converts them into the full generated sentence 
    itself

    used during training, validation, and testing/deployment
    """

    char_list = idx_to_char(pred_ids)
    char_list = tf.reshape(char_list, shape=(-1, ))
    joined_seq = tf.strings.reduce_join(char_list).numpy()
    final_seq = joined_seq.decode('UTF-8')

    return final_seq

def one_hot_encode(sparse_labels):
    """
    one hot encodes a sparse multi-class label
    e.g. if in sparse labels contain [0 1 2 0 3]
    one hot encoding would be
    [[1 0 0 0]
    [0 1 0 0]
    [0 0 1 0]
    [1 0 0 0]
    [0 0 0 1]]

    used only during training and validation
    """

    n_unique = np.unique(sparse_labels).shape[0]
    one_hot_encoding = tf.one_hot(sparse_labels, depth=n_unique)

    return one_hot_encoding

def decode_one_hot(Y_preds):
    """
    whether for image, sentiment, or general classification
    this function takes in an (m x 1) or (m x n_y) matrix of
    the predicted values of a classifier

    e.g. if binary the input Y_preds would be 
    [[0 1]
    [1 0]
    [1 0]
    [0 1]
    [1 0]
    [1 0]]

    if multi-class the Y_preds for instance would be...

    [[0 0 0 1]
    [1 0 0 0
    [0 0 1 0]
    ...
    [0 1 0 0]]

    what this function does is it takes the argmax along the
    1st dimension/axis, and once decoded would be just two
    binary categorial values e.g. 0 or 1 or if multi-class
    0, 1, 2, or 3

    main args:
        Y_preds - 

    used during training, validation, and testing/deployment
    """

    # check if Y_preds is multi-class by checking if shape
    # of matrix is (m, n_y), (m, m, n_y), or just m
    if len(Y_preds.shape) >= 2:
        # take the argmax if Y_preds are multi labeled
        sparse_categories = np.argmax(Y_preds, axis=1)

    return sparse_categories

def re_encode_sparse_labels(sparse_labels, new_labels: list=['DER', 'APR', 'NDG']):
    """
    sometimes a dataset will only have its target values 
    be sparse values such as 0, 1, 2 right at the start
    so this function re encodes these sparse values/labels
    to a more understandable representation

    upon reencoding this can be used by other encoders
    such as encode_features() which allows us to save
    the encoder to be used later on in model training

    used only during training and validation
    """

    # return use as index the sparse_labels to the new labels
    v_func = np.vectorize(lambda sparse_label: new_labels[sparse_label])
    re_encoded_labels = v_func(sparse_labels)

    return re_encoded_labels

def translate_labels(labels, translations: dict={'DER': 'Derogatory', 
                                                 'NDG': 'Non-Derogatory', 
                                                 'HOM': 'Homonym', 
                                                 'APR': 'Appropriative'}):
    """
    transforms an array of shortened versions of the
    labels e.g. array(['DER', 'NDG', 'DER', 'HOM', 'APR', 
    'DER', 'NDG', 'HOM', 'HOM', 'HOM', 'DER', 'DER', 'NDG', 
    'DER', 'HOM', 'DER', 'APR', 'APR', 'DER'] to a more lengthened
    and understandable version to potentially send back to client
    e.g. array(['DEROGATORY', NON-DEROGATORY, 'DEROGATORY', 'HOMONYM',
    'APPROPRIATIVE', ...])

    used during training, validation, and testing/deployment
    """

    v_func = np.vectorize(lambda label: translations[label])
    translated_labels = v_func(labels)
    return translated_labels

def vectorize_sent(X_trains, X_cross, X_tests):
    """
    vectorizes a set of sentences either using term frequency
    inverse document frequency or by count/frequency of a word
    in the sentence/s

    returns a vectorizer object for later saving
    """

    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_trains)
    X_trains_vec = vectorizer.transform(X_trains).toarray()
    X_cross_vec = vectorizer.transform(X_cross).toarray()
    X_tests_vec = vectorizer.transform(X_tests).toarray()
    
    return X_trains_vec, X_cross_vec, X_tests_vec, vectorizer

def normalize_ratings(ratings: pd.DataFrame):
    """
    normalizes the ratings dataframe by subtracting each original
    rating of a user to an item by the mean of all users rating
    to that item

    args: 
        ratings - a
        
    """
    # calculate mean ratings of all unique items first
    unique_item_ids = ratings['item_id'].unique()

    # build dictionary that maps unique item id's to their respective means
    items_means = {item_id: ratings.loc[ratings['item_id'] == item_id, 'rating'].mean() for item_id in unique_item_ids}

    # build list of values for mean column of new dataframe
    avg_rating = [items_means[item_id] for item_id in ratings['item_id']]

    # create avg_rating and normed_rating columns 
    temp = ratings.copy()
    temp['avg_rating'] = avg_rating
    temp['normed_rating'] = temp['rating'] - avg_rating

    # return new dataframe
    return temp

def normalize_rating_matrix(Y, R):
    """
    normalizes the ratings of user-item rating matrix Y
    note: the 1e-12 is to avoid dividing by 0 just in case
    that items aren't at all rated by any user and the sum
    of this user-item interaction matrix is not 0 which leads
    to a mathematical error.

    how this works is it takes the mean of all the user ratings
    per item, excluding of course the rating of users who've
    not yet rated the item

    args:
        Y - user-item rating matrix of (n_items x n_users) 
        dimensionality

        R - user-item interaction matrix of (n_items x n_users) 
        dimensionality
    """
    Y_mean = np.sum(Y * R, axis=1) / (np.sum(R, axis=1) + 1e-12).reshape(-1)
    Y_normed = Y - (Y_mean * R)
    return [Y_normed, Y_mean]

def encode_image(image_path: str, dimensions: tuple=(256, 256)):
    """
    encodes an image to a 3D matrix that can be used to
    feed as input to a convolutional model

    used primarily during testing/deployment to encode
    given image from client but image encoder for training
    is done by create_image_set() from loaders.py
    """

    # open image
    img = Image.open(image_path)

    # if there are given dimensions for new image size resize image
    if dimensions != None:
        img = img.resize(size=dimensions)

    # encode image by converting to numpy array
    encoded_img = np.asarray(img)

    # close image file
    img.close()

    return encoded_img


def standardize_image(encoded_img):
    """
    rescales an encoded image's values from 0 to 255 down
    to 0 and 1

    used primarily during testing/deployment to encode
    given image from client but image standardization for
    training is done by create_image_set() from loaders.py
    """

    rescaled_img = (encoded_img * 1.0) / 255

    return rescaled_img

def rejoin_batches(data_generator):
    """
    rejoins the generator batches created by a data generator
    like ImageDataGenerator and returns a numpy array
    """
    X = []
    Y = []
    batch_index = 0
    total_examples = 0

    while batch_index <= data_generator.batch_index:
        data = next(data_generator)
        images, labels = data
        total_examples += len(images)
        
        # loop through all examples in each batch
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            X.append(image)
            Y.append(label)

        batch_index = batch_index + 1
    print(total_examples)

    X_array = np.asarray(X)
    Y_array = np.asarray(Y)

    return X_array, Y_array

def activate_logits(logits):
    """
    passes the predicted logits to a softmax layer to
    obtain a probability vector that sums to 1
    """

    softmax_layer = tf.keras.layers.Activation(activation=tf.nn.softmax)
    Y_preds = softmax_layer(logits)

    return Y_preds
