import sys
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle

def load_data(database_filepath):
    """
    This function loads the database from the sqlite and outputs the various dataframe
    Input: SQLite location
    Output: X,Y,Category_names
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('ETLtable', engine)
    X = df['message'].values
    Y = df[df.columns[4:].values]
    category_names = list(Y.columns)
    return X,Y,category_names



def tokenize(text):
    """
    This function takes in the text data and gives out the list of clean tokens.
    Input: data in the form of text
    Output: list of clean tokens
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for toks in tokens:
        clean_toks = lemmatizer.lemmatize(toks).lower().strip()
        clean_tokens.append(clean_toks)
    return clean_tokens



def build_model():
    """
    This function will build the ML pipeline using random forest Classifier.

    Please note that for the gridsearch I have taken only a few parameters beacuse the
    time taken was very long and the pkl model was very huge. Please refer the
    intital preps folder in the repo and the ML pipeline notebook to view the complete
    Gridsearch.

    Input: None
    Output: GridsearchCV results
    """
    ML_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators': [100],
        'clf__estimator__min_samples_split': [2],
        'tfidf__smooth_idf': [True, False]
    }
    pipeline = GridSearchCV(ML_pipeline, param_grid=parameters, verbose=2, n_jobs = 3)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function prints the classification report
    Input: various model training parameters
    Output: None
    """
    Y_pred = model.predict(X_test)
    for i in range(36):
        print(category_names[i])
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]),
              '------------------------------------------------------------')



def save_model(model, model_filepath):
    """
    This function will export the best performing model as a pickle file
    Input: Best trained model
    Ouput: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()