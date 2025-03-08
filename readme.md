# This repository contains all generalized code snippets and templates relating to model experimentation, training, evaluation, testing, server-side loading, client-side requests, usage documentation, loaders, evaluators, visualizers, and preprocessor utilities, and the model architectures, figures, and final folder

# requirements:
1. git
2. conda
3. python

# Source code usage
1. assuming git is installed clone repository by running `git clone https://github.com/08Aristodemus24/<repo name>`
2. assuming conda is also installed run `conda create -n <environment name e.g. some-environment-name> python=x.x.x`. Note python version should be `x.x.x` for the to be created conda environment to avoid dependency/package incompatibility.
3. run `conda activate <environment name used>` or `activate <environment name used>`.
4. run `conda list -e` to see list of installed packages. If pip is not yet installed run conda install pip, otherwise skip this step and move to step 5.
5. navigate to directory containing the `requirements.txt` file.
5. run `pip install -r requirements.txt` inside the directory containing the `requirements.txt` file
6. after installing packages/dependencies run `python index.py` while in this directory to run app locally

# App usage:
1. control panel of app will have 3 inputs: prompt, temperature, and sequence length. Prompt can be understood as the starting point in which our model will append certain words during generation for instance if the prompt given is "jordan" then model might generate "jordan is a country in the middle east" and so on. Temperature input can be understood as "how much the do you want the model to generate diverse sequences or words?" e.g. if a diversity of 2 (this is the max value for diversity/temperature by the way) then then the model might potentially generate incomprehensible words (almost made up words) e.g. "jordan djanna sounlava kianpo". And lastly Sequence Length is how long do you want the generated sequence to be in terms of character length for isntance if sequence length is 10 then generated sequence would be "jordan is."

# File structure:
```
|- client-side
    |- public
    |- src
        |- assets
            |- mediafiles
        |- boards
            |- *.png/jpg/jpeg/gig
        |- components
            |- *.svelte/jsx
        |- App.svelte/jsx
        |- index.css
        |- main.js
        |- vite-env.d.ts
    |- index.html
    |- package.json
    |- package-lock.json
    |- ...
|- server-side
    |- modelling
        |- data
        |- figures & images
            |- *.png/jpg/jpeg/gif
        |- final
            |- misc
            |- models
            |- weights
        |- metrics
            |- __init__.py
            |- custom.py
        |- models
            |- __init__.py
            |- arcs.py
        |- research papers & articles
            |- *.pdf
        |- saved
            |- misc
            |- models
            |- weights
        |- utilities
            |- __init__.py
            |- loaders.py
            |- preprocessors.py
            |- visualizers.py
        |- __init__.py
        |- experimentation.ipynb
        |- testing.ipynb
        |- training.ipynb
    |- static
        |- assets
            |- *.js
            |- *.css
        |- index.html
    |- index.py
    |- server.py
    |- requirements.txt
|- demo-video.mp5
|- .gitignore
|- readme.md
```

# Articles:
1. multiple/ensemble model training: 
* https://www.geeksforgeeks.org/lazy-predict-library-in-python-for-machine-learning/
* https://medium.com/omics-diary/how-to-use-the-lazy-predict-library-to-select-the-best-machine-learning-model-65378bf4568e

2. evaluating ensemble models:

# Pre-built template functions for projects:
~ - done
^ - needs further testing
& - needs further understanding
! - needs further tweaking

Loaders:
NLP
* load_corpus
* get_chars
* construct_embedding_dict
* construct_embedding_matrix

General
* load_lookup_array
* save_lookup_array
* load_meta_data
* save_meta_data
* get_cat_cols
* get_top_models
* load_model
* save_model
* create_metrics_df ~
* create_classified_df ~

Image Processing
* create_image_set

Preprocessors:
NLP
* map_value_to_index
* remove_contractions
* rem_non_alpha_num
* capitalize
* filter_valid
* partition_corpus
* rem_stop_words
* stem_corpus_words
* lemmatize_corpus_words
* string_list_to_list
* flatten_series_of_lists
* sentences_to_avgs
* init_sequences
* decode_id_sequences
* decode_one_hot ~

Recommendation
* normalize_ratings
* normalize_rating_matrix

General
* normalize_train_cross
* encode_features ~
* translate_labels ~

Visualizers:
* plot_train_cross_features ~
* analyze ~ 
* view_words
* data_split_metric_values ~
* view_value_frequency ! has no x label
* multi_class_heatmap ~
* view_metric_values ~
* view_classified_labels ~
* view_label_freq ! has no x label
* disp_cat_feat ! has x labels but bars are too compressed especially if n unique is large
* describe_col
* visualize_graph
* plot_evolution ~
* view_clusters_3d ~
* ModelResults ! since there might be a shorter version of code for it from the micro-organism-classifier kaggle code
* plot_all_vars ~

# Prebuilt template components for client-side:
* WE NEED TO IMPLEMENT NOW THESE TEMPLATES FOR OUR MICRO-ORGANISM-CLASSIFIER
* <s>submit button ~</s>
* <s>100vh section for landing page ~</s>
* <s>text input fields ~</s>
* <s>range input fields ~</s>
* <s>number input fields ~</s>
* <s>file input fields ~</s>
* <s>select fields ~</s>
* <s>textarea fields ~</s>
* <s>fix width of data-form-content why does it not expand the width of the parent? ~ it was because of its parent data-form-section being set to display flex and shrinking it to the width of childrens size, or in other words in auto</s>
* <s>fix image upload field spacing ~</s>
* <s>fix image upload fields upload button ~</s>
* <s>create custom themed classes to switch between themes for components</s>
* <s>implement dark</s>
* <s>tweak light shadow of light theme</s>
* fix image upload fields image height ! needs to be responsive from 1600px to 320px
* copy svelte component templates and write it in react
* embed alert instead in form instead of it being a separate component to form

# Prebuilt template functions for server-side
* <s>for general models</s>
* <s>for tensorflow models</s>
* using a pipeline for preprocessing user input from client-side using 
a preprocessing function e.g. featture vector -> normalizer loaded with 
specific mean and standard deviation, do I use a saved .json object with
these hyper params, or use a sklearn object instead? But a sklearn object 
like a normalizer I cannot save

```
def predict(self, X):
        # normalize on training mean and standard dev first
        X = (X - self.mean) / self.std_dev

        # predict then return prediction value
        return self.linear(X)
```

<s>use instead your own implementation of a normalizer function and then loading
the respective hyper params like mean and standard deviation to pass into 
this from scratch implementation of a normalizer</s>

<s>but not only for an a normalizer, what about for OrdinalEncoder() and 
LabelEncoder() objects? I really can't save them because it would just be too much</s>

<s>if there is a dataset X and it has categorical variables
does we really need to save the encoder we used on this dataset to use on the</s>

X_train
yes, bacteria
no, archaea
no, eukarya
yes, eukarya
no, bacteria

X_cross
no, bacteria
yes, eukarya

<s>AH YES. we really need to save the encoder or find some way to use the encoders
information that was obtained from the whole dataset (since we do
not split the data set if we encode the features because potential categories
of features may be lost on splitting the data) because if we use a new encoder 
on the cross dataset the features may be such that some would be missing that 
would exist in the train data, e.g. bacteria and archaea are the only features 
so we don't want to encode it as only 0 and 1 since and in the whole data there 
are 3 categories for a feature nsmrly bacteria, archaea, and eukarya which are 
encoded to 0</s>

**or just save the sklearn encoders as well** using save_model() and load it using load_model()


* <s>there needs to be also a meta_data saver like this
```
def save_weights(self):
    meta_data = {
        'non-bias': self.theta.tolist(),
        'bias': self.beta.tolist()[0],
        'mean': self.mean.tolist(),
        'std_dev': self.std_dev.tolist()
    }

    # if directory weights does not already exist create 
    # directory and save weights there
    if os.path.exists('./weights') != True:
        os.mkdir('./weights')
    
    with open('./weights/meta_data.json', 'w') as out_file:
        json.dump(meta_data, out_file)
        out_file.close()
```
</s>

for both general and deep learning models. But since sklearn models can be saved directly without taking into account its hyperparams, we could only now save certain other hyperparams like again mean and standard deviation

or in tensorflow models some meta_data may be important to load later on such as:
```
# global variables
vocab = None
char_to_idx = None
idx_to_char = None
hyper_params = None
```
just in case a model only saves its weights, and not the whole model architecture such as hyper_params in this case

but even if the architecture itself is saved its respective hyper params, for preprocessing input in order to feed to the trained model it is important that such hyperparams used in preprocessing the input for having even trained the model in the first place is important, this being vocab, char_to_idx, and idx_to_char for instance

* <s>writing decoders functions for the predictions of our model
- decoder for categorical data which can range from:
a. categorical vectors to image class
b. categorical sequences of vectors to sentences
c. categorical vectors to sentiments/emotional reactions class (such is the case for NLP)</s>

* <s>write preprocessor for when user inputs an image from the client-side</s>
* writing dropout for CNNs https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/


# Tuner template for image processing model
* only thing that is left now is to build a tuner for this
* then figure out a way to upload the these modules so I can access these models on a notebook on collab easily
* also how do I encode labels for each category? No wait it is already done by the create_image_set() loader


# Instructions for loading, preprocessing, and visualizing data as well as training and tuning models
* clone project-seraphim
* collab is where you will import only the basic libraries for reading and splitting data since data is highly variable and cannot be loaded by a single function
* delete uneccessary libraries and cells in experimentation.ipynb
* delete uneccessary libraries and functions in loaders.py
* delete uneccessary libraries and functions in preprocessors.py
* delete uneccessary libraries and functions in visualizers.py
* collab is where I will load, preprocess, and visualize data using loaders, processors, and visualizers
* collab is where I will train the baseline
* collab is where I will save the baseline
* collab is where I will serach optimal hyperparams of model
* collab is where I will train tuned model with optimal hyperparams
* collab is where I will save the tuned model



# Insights
now I understand
lambda
n_rnn_units
tuner/epochs
tuner/initial_epoch
tuner/bracket
are all hyperparameters
that is why we can have a max epoch parameter for kt.HyperBand that means the number of epochs to find the best for example accuracy can be at minimum 1 and at maximum the value of max epoch we have defined

but my question is if we already have max epochs for e.g. 1 epoch for model 1 with 0.1 lambda, 0.5 dropout, and 2 epoch

why do we still have in tuner.search a set number of epochs which is higher than the max epochs of the kt.Hyperband object


maybe max_epochs is just really meant to be the MAXIMUM epochs tuner.search would go sincei t is after all a hyper param that needs to be tuned only in this RANGE

I have 30 trials over all so maybe the max epochs 10 and factor 3 is multiplied to trial 30 times different hyper params

The algorithm trains a large number of models for a few epochs e.g. 1 epoch for the first trial and carries forward only the top-performing half of models to the next round/trial/bracket

Hyperband determines the number of models to train in a bracket by computing 1 + logfactor(max_epochs) and rounding it up to the nearest integer.

so if factor is 3 and the max_epochs is 10 then $log_{3}(10)$ would be 2.09590 and when added to 1 is 3.09590 which when rounded is 3

so if we have 3 models per round/trial/bracket and we take only the top performing half of these 3 models

for maxepochs of 5 and factor of 3 the number of trials were 10

* it is better to tune a model first in colab to not waste gpu resources and to determine as fast as possible the best model with the best hyper parameters

* deleting latest commit or any commit can be done by `git reset --hard HEAD~<nth commit e.g. 1 (means latest commit 2 means second latest commit)> to delete latest commit`
* pushing new files that conflicts with current master branch can be forced by `git push origin <your_branch_name> --force`

* https://stackoverflow.com/questions/55627884/react-fetch-api-getting-415-unsupported-media-type-using-post
* https://dev.to/brunooliveira/uploading-a-file-svelte-form-and-springboot-backend-18m6
* https://muffinman.io/blog/uploading-files-using-fetch-multipart-form-data/
* https://stackoverflow.com/questions/35192841/how-do-i-post-with-multipart-form-data-using-fetch