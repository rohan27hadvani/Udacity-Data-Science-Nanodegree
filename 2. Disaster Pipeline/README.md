# Disaster Response pipeline Project
## Description
In this project I have build a model to classify the messages sent during a disaster. It contains the real messages provided by Figure - eight.

I have created two pipelines 1 for ETL & the other one for ML.

I have also created a WebApp which allows us to classify a message and view some of the data analysis.

## File Description

    .
    ├── app     
    │   ├── run.py                           # Flask file that runs app
    │   └── templates   
    │       ├── go.html                      # Classification result page
    │       └── master.html                  # Main page of web app    
    ├── data                   
    │   ├── disaster_categories.csv          # Dataset including all the categories  
    │   ├── disaster_messages.csv            # Dataset including all the messages
    │   ├──process_data.py                   # Data cleaning
    └   └──DisasterResponse.db               # Database containing Cleaned data
    ├── models
    │   └── train_classifier.py              # Train ML model           
    └── README.md

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
