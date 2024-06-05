"""
Preprocessing of Data
Project: Disaster Response Pipeline (Udacity - Data Scientist 2)

Execution Syntax:
> python process_data.py <path/to/csv/containing/messages> <path/to/csv/containing/categories> <path/to/sqllite/db>
> OR 
> python3 process_data.py <path/to/csv/containing/messages> <path/to/csv/containing/categories> <path/to/sqllite/db>

Arguments :
    1) Path to the CSV file containing messages (e.g. disaster_messages.csv)
    2) Path to the CSV file containing categories (e.g. disaster_categories.csv)
    3) Path to SQLite destination database (e.g. disaster_response_db.db)
"""
# Importing libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_csv(message_path, categories_path):
    df_mess = pd.read_csv(message_path)
    df_cat = pd.read_csv(categories_path)
    df = pd.merge(df_mess, df_cat, on='id')
    return df

def clean_data(df):
    categories = df['categories'].str.split(pat=";",expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis =1)
    df = df.drop_duplicates()
    return df

def save_data_to_db(df, database_filename):
    """
    Save Data to SQLite Database Function
    
    Arguments:
        df -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite destination database
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')

def main():
    """
    Main function starts the sequence of functions. There are three primary actions taken by this function:
        - Load Data from .csv files
        - Clean categories data
        - Save data to SQLite database
    """
    # Execute the ETL pipeline if the count of arguments is matching to 4
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:] # Extract the parameters in relevant variable

        print('Loading messages data from {} ...\nLoading categories data from {} ...'
              .format(messages_filepath, categories_filepath))
        
        df = load_csv(messages_filepath, categories_filepath)

        print('Cleaning categories data ...')
        df = clean_data(df)
        
        print('Saving data to SQLite DB : {}'.format(database_filepath))
        save_data_to_db(df, database_filepath)
        
        print('Cleaned data has been saved to database!')
    
    else: # Print the error message 
        print("""Please provide the arguments correctly: \nSample Script Execution:\n\
                Execution Syntax: \n 
                >python process_data.py <path/to/csv/containing/messages> <path/to/csv/containing/categories> 
                <path/to/sqllite/db> \n 
                > OR \n> python3 process_data.py <path/to/csv/containing/messages> <path/to/csv/containing/categories> 
                <path/to/sqllite/db>""")
              

if __name__ == '__main__':
    main()
