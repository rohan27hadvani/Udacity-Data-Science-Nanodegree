import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function reads the 2 data and merges them into 1 single dataframe
    Input: messages_filepath, categories_filepath
    Output: Merged dataframe df
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id', how='inner')
    return df

def clean_data(df):
    """
    This function takes the merged dataframe df from the previous function and cleans it for further analysis
    Input: Merged df
    Output: Cleaned df
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda x: x.split('-')[0]).values.tolist()
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    df.head()
    return df


def save_data(df, database_filepath):
    """
    This function takes the cleaned dataframe and stores it in the form of a sqlite database
    Input: Cleaned df
    output: SQLite database
    """
    engine = create_engine('sqlite:///' +database_filepath)
    df.to_sql('ETLtable', engine, index=False)



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()