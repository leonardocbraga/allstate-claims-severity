import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import model

print("Loading...")
train_df = model.load_data()
test_df = model.load_data(test = True)

split = train_df.shape[0]
ids = test_df.iloc[0::, 0]
X_df = pd.concat([train_df.iloc[0::, 1:-1], test_df.iloc[0::, 1::]])
y_df = train_df.iloc[0::, -1]
del train_df, test_df

print("Preprocessing...")
X_joined, y_train = model.pre_process(X_df, y_df)

X_train = X_joined[0:split]
X_test = X_joined[split::]
del X_df, y_df, X_joined

print("Fitting...")
clf = model.fit(X_train, y_train)

print("Predicting...")
predicted = clf.predict(X_test)
output = model.pre_process_y(predicted, reverse = True)

print("Creating file...")
submission = pd.DataFrame()
submission['loss'] = output
submission['id'] = ids
submission.to_csv('submission.csv', index=False)
print('Done.')

del X_train, X_test, y_train, predicted
del output, clf
