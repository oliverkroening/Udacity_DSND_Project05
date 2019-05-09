# import libraries
import pandas as pd
import numpy as np
import re
from time import time
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(data_file, table_name):
    '''
    Function:   load data from SQLite database and return X and y variables
    Args:
        data_file (str):    path of SQLite database
        table_name (str):   name of table in SQLite database
    Return:
        X (pd.DataFrame):   messages (features)
        y (pd.DataFrame):   categories (labels)
    '''
    # read in file



    # clean data


    # load to database


    # define features and label arrays


    return X, y


def build_model():
    # text processing and model pipeline


    # define parameters for GridSearchCV


    # create gridsearch object and return as final model pipeline


    return model_pipeline


def train(X, y, model):
    # train test split


    # fit model


    # output model test results


    return model


def export_model(model):
    # Export model as a pickle file



def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline
