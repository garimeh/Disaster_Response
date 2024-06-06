"""
Classifier Trainer
Project: Disaster Response Pipeline (Udacity - Data Science Nanodegree)

Sample terminal script:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>

Arguments:
    1) Path to SQLite destination database (e.g. disaster_response_db.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)
"""

# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pandas as pd
pd.set_option('display.max_columns', 100)
import os
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sqlalchemy import create_engine
import sys


def load_data(database_filepath):
    """
    Load Data from the Database Function
    
    Arguments:
        database_filepath -> Path to SQLite destination database (e.g. disaster_response_db.db)
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    X = df['message']
    y = df.iloc[:,4:]

    category_names = y.columns 
    return X, y, category_names

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model(X, y, clf = RandomForestClassifier()):
    """
    INPUT:
    clf = classifier model (If none is inputted, the function will use default 'RandomForestClassifier' model) 
    
    OUTPUT:
    cv = ML model pipeline after performing grid search
    """
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(clf))
        ])
    
    parameters = {
    'vect__max_df': [0.75, 1.0],
    'vect__max_features': [None, 1000],
    'tfidf__use_idf': [True, False],
    'clf__estimator__n_estimators': [50, 70],
    'clf__estimator__min_samples_split': [2, 3],
    }
        
    cv = GridSearchCV(pipeline, parameters, cv=2, verbose=3, n_jobs=-1) 
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """
    INPUT:
    model - ML model
    X_test - test messages
    y_test - categories for test messages
    category_names - category name for y
    
    OUTPUT:
    none - print scores (precision, recall, f1-score) for each output category of the dataset.
    """
    y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        print(column)
        print(classification_report(y_test.iloc[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    """
    INPUT:
    model - ML model
    model_filepath - location to save the model
    
    OUTPUT:
    none
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """
    Starts the sequence for model training.
    """
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
