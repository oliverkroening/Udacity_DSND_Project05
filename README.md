# Udacity Data Science Nanodegree
# Project 05: Disaster Response Pipelines

--------------------------------------
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## 1. Installation <a name="installation"></a>

- This code was created by using Python versions 3.*.
- Following libraries are required:

* sys
* pandas
* numpy
* matplotlib
* seaborn
* re (regular expressions)
* sqlalchemy
* nltk
* pickle
* scikit-learn (sklearn)

- copy repository: git clone https://github.com/oliverkroening/Udacity_DSND_Project05


## 2. Project Motivation <a name="motivation"></a>
For this Udacity project, I used data from [Figure Eight](https://www.figure-eight.com/) to perform data engineering tasks. The goal is to create an ETL pipeline in which I extract messages that were sent during disaster events and their categories. After that, cleaning and transformation tasks are performed to reduce complexity of the text messages and to simplify the assignment of categories. The merged data is then stored in an SQL database.

The second goal is to create a Machine Learning Pipeline in which a classifier is built, trained and evaluated. This classifiers is able to assign messages to one or more of the defined 36 categories. 

First, the pipelines are created and tested within Jupyter Notebooks. These notebooks are only for preparation of the .py files that are callable by terminal commands (see [File Descriptions](#files)). At the end, a web application is designed using Flask to visualize different plots regarding disaster response messages.

## 3. File Descriptions <a name="files"></a>  
The project consists of four folders:
- app
  - template
    - `master.html` (main page of web app)
    - `go.html`(classification result page of web app)
  - `run.py` (Flask file that runs app)

- data
  - `disaster_categories.csv` (categories database, that has to be processed) 
  - `disaster_messages.csv`(messages database, that has to be processed)
  - `process_data.py` (ETL pipeline)
  - `DisasterResponse.db` (database of save clean data)

- models
  - `train_classifier.py` (ML pipeline)
  - `disaster_response_clf.pkl` (saved model)
  - `disaster_response_clf1.pkl`  (saved model)

The ETL pipeline can be called in a terminal as follows:
python process_data.py <path_to_messages_database> <path_to_categories_database> <SQL_database_destination>
Example:
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

The ML pipeline can be called in a terminal as follows:
python train_classifier.py <path_to_SQL_database> <classifier_destination>
Example:
python train_classifier.py ../data/DisasterResponse.db disaster_response_clf.pkl

To run the web application, you have to type in the command line:
python run.py
If there are no errors, you can open another terminal window and type
env|grep WORK
to show the WORKSPACEDOMAIN and the WORKSPACEID. After that, you have to open a browser window to open the app on port 3001:
https://SPACEID-3001.SPACEDOMAIN

## 4. Results <a name="results"></a>
Two classifiers were built, trained and evaluated during this project. Both show quite good results in classifying messages with an f1-score of around 0.67.

## 5. Licensing, Authors, Acknowledgements<a name="licensing"></a>
All data was provided by Figure Eight and Udacity, thus, I must give credit to them. Other references are cited in the Jupyter notebook.
Please refer to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.
