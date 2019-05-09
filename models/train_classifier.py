# import libraries
import sys
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
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(db_path = '../data/DisasterResponse.db', table_name = "messages"):
    '''
    Function:   load data from SQLite database and return X and y variables
    Args:
        data_file (str):    path of SQLite database
        table_name (str):   name of table in SQLite database
    Return:
        X (pd.DataFrame):   messages (features)
        y (pd.DataFrame):   categories (labels)
    '''
    engine = create_engine('sqlite:///'+db_path)
    df = pd.read_sql_table(table_name, engine)
    X = df.message
    y = df.drop(['id','message','genre','original'], axis=1)
    return X, y

def tokenize(text):
    """
    Function: tokenize text
    Args:
        text (str): string to tokenize
    Return:
        clean_tokens (list of str): list of cleaned and tokenized strings
    """
    # 1. normalization
    # remove punctuation and capitalization of words
    text = re.sub(r'[^a-zA-Z0-9]', " ", text.lower())

    # 2. tokenization
    # tokenize words from text
    tokens = word_tokenize(text)

    # 3. stop word removal
    # remove all words that can be found in the stopwords dictionary
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # 4. stemming and lemmtization
    # define lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Function: create GridSearch model (TF-IDF-Vectorizer, AdaBoostClassifier)
    Args:
        None
    Return:
        cv (GridSearch): GridSearchObject
    """
    # create pipeline
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # define parameters for GridSearch
    parameters = {
        'vect__max_df': [0.9, 1.0],
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__learning_rate':[0.9, 1.0]
    }

    # create model
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=1)

    return cv

def evaluate_model(model, X_test, y_test):
    """
    Function: predict output of model, evaluates result against test dataset
    Args:
        model (classifier obj): trained classifier model
        X_test (pd.DataFrame): features of testing dataset
        y_test (pd.DataFrame): labels of testing dataset
    Return:
       None
    """
    # make predictions
    y_pred = model.predict(X_test)

    # create classification report
    print(classification_report(y_test, y_pred,
                        target_names=list(y_test.columns)))

    # print total accuracy, precision, recall on testing dataset
    print('accuracy:\t{:.4}'.format(accuracy_score(y_test,y_pred)))
    print('precision:\t{:.4}'.format(precision_score(y_test,y_pred,
                                                average='weighted')))
    print('recall:\t\t{:.4}'.format(recall_score(y_test,y_pred,
                                                average='weighted')))

def export_model(model, clf_path = '../models/disaster_response_clf.pkl'):
    """
    Function: save model as pickle file
    Args:
        model (GridSearch obj): trained and tuned classifier model
        clf_path (str): path of pickle file destination
    Return:
        None
    """
    with open(clf_path, 'wb') as f:
        pickle.dump(model, f)


def import_model(clf_path = '../models/disaster_response_clf.pkl'):
    """
    Function: load model from pickle file
    Args:
        clf_path (str): path of pickle file
    Return:
        model (GridSearch obj): loaded model
    """
    with open(clf_path, 'rb') as f:
        model = pickle.load(f)

    return model

def main():
    """
    Function: load, train, evaluate and save a ML model
    Args:
        0:  database_path (str)
        1:  database_table (str)
        2:  model_path (str)
    Return:
        None
    """
    if len(sys.argv) == 3:
        database_path = sys.argv[1]
        model_path = sys.argv[2]

        print("Loading data...")
        X,y = load_data(db_path=database_path)
        print("Data loaded!")
        print("------------")

        print("Building model...")
        model = build_model()
        print("Model built!")
        print("------------")

        print("Training model...")
        start = time()
        X_train, X_test, y_train, y_test = train_test_split(X,y,
                                            test_size = 0.20, random_state=42)
        model.fit(X_train, y_train)
        print("Training Time: {:.0f} seconds".format(time()-start))
        print("Model trained!")
        print("------------")

        print("Evaluating model...")
        evaluate_model(model.best_estimator_, X_test, y_test)
        print("Model evaluated!")
        print("------------")

        print("Saving model...")
        export_model(model, model_path)
        print("Model saved at ", clf_path, "!")
        print("------------")

    else:
        print("Please provide the filepaths of the database (first argument) "\
              "and the destination for the model export (second argument)!")

if __name__== "__main__":
    main()
