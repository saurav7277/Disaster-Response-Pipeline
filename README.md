### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

1-Anaconda distribution of python

2-punkt,stopwords,wordnet frpm nltk library

3-python 3.0

## Project Motivation<a name="motivation"></a>

For this project, I was interested in building an ETL pipeline that will load a cleaned data into a sql database and then a Machine learning pipeline that will create a model which will be able to classify disaster response messages coming from various sources into 36 different categories. 


## File Descriptions <a name="files"></a>

1-data/disaster_messages.csv :- dataset containing disaster response messages

2-data/disaster_categories.csv :-dataset containing disaster categories

3-data/process_data.py :- ETL pipeline to extract data,transfrom data and load data into sql database

4-model/train_classifier.py :- Machine learning pipeline trained on cleaned dataset,able to classify disaster response messages into 36   different categories

5-app/go.html,master.html :-starter file

6-app/run.py :- python script to run flask web app

## Results<a name="results"></a>

ETL pipeline perfectly able to extract,transform and load cleaned dataset into a sql database.

Machine learning pipeline almost classify disaster response messages into different categories perfectly.

Flask web app shows data visualisation on training data and provide gui for classifying disaster response messages
## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Author-Saurav Kumar
