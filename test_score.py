import numpy as np
import pandas as pd
import model

from sklearn import metrics
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, boxcox


print("Loading...")
train_df = model.load_data()
X_df = train_df.iloc[0::, 1:-1]
y_df = train_df.iloc[0::, -1]
del train_df

print("Preprocessing...")
X, y = model.pre_process(X_df, y_df)
del X_df, y_df

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.1, random_state = 0)
del X, y

print("Fitting...")
clf = model.fit(X_train, y_train)

print("Predicting...")
predicted = clf.predict(X_test)
output = model.pre_process_y(predicted, reverse = True)
del predicted

print("Training set score: %f" % clf.score(X_train, y_train))
print("Test set score: %f" % clf.score(X_test, y_test))
#print("Mean absolute error: %f" %  (sum(abs(output-(np.exp(y_test) - shift))) / output.shape[0]))
print("Mean absolute error: %f" %  mean_absolute_error(model.pre_process_y(y_test, reverse = True), output))

del X_train, X_test, y_train, y_test
del output, clf
