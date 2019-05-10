import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_path = "messages.csv", categories_path = "categories.csv"):
    '''
    Function:   load and merge message and categorie databases from csv files
    Args:
        message_path (str): path of message database (CSV-file)
        categories_path (str): path of categories database (CSV-file)
    Return:
        df (pd.DataFrame):  created DataFrame
    '''
    # load messages dataset
    messages = pd.read_csv(messages_path)
    # load categories dataset
    categories = pd.read_csv(categories_path)
    # merge datasets
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    '''
    Function:   perform one-hot encoding and clean DataFrame
    Args:
        df (pd.DataFrame):  raw DataFrame
    Return:
        df (pd.DataFrame):  cleaned DataFrame
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    # extract a list of new column names for categories
    category_colnames = categories.loc[0,:].apply(lambda x: x[:-2]).values.tolist()
    # rename the columns of the categories dataframe
    categories.columns = category_colnames
    # convert categories to numbers 0 and 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        # set values > 1 to 1 to obtain a one-hot-encoded categories table
        categories.loc[categories[column] > 1, column] = 1
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    '''
    Function:   save DataFrame as SQLite database
    Args:
        df (pd.DataFrame):  cleaned DataFrame
        database_filename (str): path of database destination
    Return:
        None
    '''
    # save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql("messages", con=engine, index=False, if_exists="replace")

def main():
    '''
    Function:   load and clean data and store into SQLite database
    Args:
        [0]:    datapath to messages CSV-file
        [1]:    datapath to categories CSV-file
        [2]:    path of database destination
    Return:
        None
    '''
    if len(sys.argv) == 4:
        messages_path = sys.argv[1]
        categories_path =  sys.argv[2]
        database_filename =  sys.argv[3]
        df = load_data(messages_path, categories_path)
        df = clean_data(df)
        save_data(df, database_filename)
    else:
        print("Please provide a path to the message CSV-file (first argument),"\
              "a path to the categories CSV-file (second argument) and" \
              "a path for the SQLite database destination (third argument).")

if __name__ == '__main__':
    main()
