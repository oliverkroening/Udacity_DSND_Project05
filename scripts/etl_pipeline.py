import pandas as pd
from sqlalchemy import create_engine

def etl_pipeline(db_path, messages_path = "../data/messages.csv", categories_path = "../data/categories.csv", table_name ="messages_tab"):
    '''
    Function: load and merge message and categorie databases from csv files, perform one-hot encoding, save in SQLite database
    Args:
        db_path (str): path of SQLite database
        message_path (str): path of message database (CSV-file)
        categories_path (str): path of categories database (CSV-file)
        table_name (str): name of table in SQLite database
    Return:
        None
    '''

    # load messages dataset
    messages = pd.read_csv(messages_path)

    # load categories dataset
    categories = pd.read_csv(categories_path)

    # merge datasets
    df = messages.merge(categories, on='id')

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

    # save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///' + db_path)
    df.to_sql(table_name, engine, index=False)
