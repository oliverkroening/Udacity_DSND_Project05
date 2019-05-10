import json
import plotly
import pandas as pd
from collections import Counter
import re
from pathlib import Path

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

def count_words(df):
    '''
    Function:   counts all tokens and returns the top 25 words,
                since function requires a lot of processing time,
                we can load the output from an existing text-file.
    Args:
        df (pd.DataFrame): DataFrame containing all messages
    Return:
        top_words_names (list of str): top 25 words
        top_words_counts (list of int): frequency of top 25 words
    '''
    words_file = Path("../data/top_words.txt")
    if words_file.is_file():
        with open(words_file) as f:
            top_words = f.readlines()
        top_words = [t.replace("\n","").replace("(","").replace(")","").replace(
                        " ","").replace("'","").split(",") for t in top_words]
        top_words = [tuple([t[0],int(t[1])]) for t in top_words]
    else:
        # create Counter object
        words = Counter()
        # iterate through messages in DataFrame
        for message in df.message:
            # tokenize each message
            tokens = tokenize(message)
            # iterate through created tokens
            for token in tokens:
                words[token] += 1
        # filter top 25 words
        top_words = words.most_common(25)
    top_words_names = [w[0] for w in top_words]
    top_words_counts = [w[1] for w in top_words]
    return top_words_names, top_words_counts

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/disaster_response_clf.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # count messages grouped by genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # count messages for each category
    cat_counts = df.iloc[:,4:].sum()
    cat_names = list(cat_counts.index)

    # count top 25 words and frequency
    top_words_names, top_words_counts = count_words(df)


    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_words_names,
                    y=top_words_counts
                )
            ],

            'layout': {
                'title': 'Most frequent Words in all Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
