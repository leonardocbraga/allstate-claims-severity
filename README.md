# Project

This task refers to the [Kaggle competition](https://www.kaggle.com/c/allstate-claims-severity/) developed by [Allstate](https://www.allstate.com/) to accurately predict claims severity. It consists of files to build the model responsible for loading data from training and test set, preprocessing, fitting and predicting. Also, there is a file to test how well the classifier chosen is doing on 10% of the training set after fitting the other 90%, and a file to plot the skewness of the data.

# Platform

The source code is written in [Python] (https://www.python.org/) with [scikit-learn] (http://scikit-learn.org/) once they can afford Machine Learning Algorithms.

# Data
The training and test sets can be obtained at [the challenge data page](https://www.kaggle.com/c/allstate-claims-severity/data/). The training and the test set are lists in which each row in these datasets represents an insurance claim, and the value to predict is called 'loss'.

# Solution

The classifier used to fit the data was [XGBRegressor](http://xgboost.readthedocs.io/en/latest/python/python_api.html) from XGBoost package. This is a scalable and flexible Gradient Boosting to train examples with 130 features, with variables labeled as 'cat' are categorical, while those labeled as 'cont' are continuous.

# Score

Getting it done gives a score of 1145.85990 on the [Public Leaderboard](https://www.kaggle.com/c/allstate-claims-severity/leaderboard/). This is the mean absolute error calculated on test set.
