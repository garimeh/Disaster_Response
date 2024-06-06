
# Disaster Response Pipeline

This Project is part of Data Scientist 2 Nanodegree Program by Udacity in collaboration with Appen. There are 2 datasets being used: 
- messages.csv contains labelled tweet and messages from   real-life disaster events. 
- categories.csv contains the categories in which the tweets are to be labelled.

The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

This project is divided in the following key sections:

1. Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB.

2. Build a machine learning pipeline to train the which can classify text message in various categories.

3. Run a web app which can show model results in real time

## Important Dependencies
To run this project you need to clone this repository and install the following libraries in your python virtual environment.

Use this command to make your virtual environment:
```
    python -m venv <name of virtualenv>
```
Then, use ```pip install -r requirements.txt``` to install the following libraries.

- Python 3.10 and above.
- Sciki-Learn
- NLTK
- SQLalchemy
- Pickle
-  Flask
- Plotly

## Executing Program:
You can run the following commands in the project's directory to set up the database, train model and save the model.

- To run ETL pipeline to clean data and store the processed data in the database
  ```python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db```
- To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file ```python train_classifier.py data/DisasterResponse.db model.pkl```
- Run the following command in the app's directory to run your web app:
    
       ```python run.py```

- Go to http://0.0.0.0:3001/

## Note:
- I have not been able to add my trained model due to the memory limits for each file on github. My model was coming out to be more than 100MB, which is the file limit on github.
- When dealing with an imbalanced dataset, especially in tasks like multi-label classification for disaster response, certain categories can have significantly fewer samples than others. This imbalance affects the training and performance evaluation of machine learning models in several key ways:
    - **Precision**: This is the ratio of true positives to the sum of true and false positives. It indicates the accuracy of the positive predictions. In the context of an imbalanced dataset, high precision means that when the model predicts a class, it is likely correct, which is crucial for certain disaster response categories where false alarms (false positives) can be costly or dangerous.
    -  **Recall**: Also known as sensitivity, it is the ratio of true positives to the sum of true positives and false negatives. High recall is particularly important in disaster response scenarios, as it reflects the model's ability to capture all relevant instances (e.g., identifying all urgent distress messages). In imbalanced datasets, recall becomes critical for the minority class because failing to identify true positives can have serious consequences.

### Acknowledgements
Udacity for providing an amazing Data Scientist Nanodegree Program
