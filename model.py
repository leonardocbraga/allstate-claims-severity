import numpy as np
import pandas as pd

from scipy.stats import skew, boxcox

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

TRAINING_FILE_NAME = '../input/train.csv'
TEST_FILE_NAME = '../input/test.csv'

def load_data(test = False):
    data_frame = pd.read_csv(TRAINING_FILE_NAME if not test else TEST_FILE_NAME, header=0)

    return data_frame

def get_encoders(data_frame):
    split = len(data_frame.columns[data_frame.columns.map(lambda column: column.startswith('cat'))])

    encoders = dict()
    for col in data_frame.columns[0:split].values:
        cats = np.unique(data_frame[col])

        label_encoder = LabelEncoder()
        label_encoder.fit(cats)

        onehot_encoder = OneHotEncoder(sparse=False, n_values=len(cats))

        encoders[col] = {'label': label_encoder, 'onehot': onehot_encoder}

    return encoders

def encode_label(data_frame, encoders):
    split = len(data_frame.columns[data_frame.columns.map(lambda column: column.startswith('cat'))])
    for i in range(0, split):
        col = data_frame.columns[i]

        label_enc = encoders[col]['label']

        data_frame[col] = label_enc.transform(data_frame[col])

def encode_onehot(X, encoders):
    features = []
    columns = X.columns[X.columns.map(lambda column: column.startswith('cat'))]
    new_split = len(columns)
    for i in range(0, new_split):
        col = columns[i]

        transformed = X[col].values
        transformedReshaped = transformed.reshape(X.shape[0], 1)

        onehot_encoder = encoders[col]['onehot']
        transformedOneHot = onehot_encoder.fit_transform(transformedReshaped)
        features.append(transformedOneHot)

        del transformed
        del transformedReshaped
        del transformedOneHot

    new_features = np.column_stack(features)
    del features

    for i in range(new_split, X.shape[1]):
        col = X.columns[i]

        X[col] = boxcox(X[col] + 1)[0]

    result = np.concatenate((new_features, X.iloc[0::, new_split::].values), axis = 1)
    del new_features
    return result

def pre_process_onehot(X, y, encoders):
    encode_label(X, encoders)

    X_result = encode_onehot(X, encoders)

    return X_result

def get_AZ_labels():
    labels = np.empty(0)
    for l in range(ord('A'), ord('Z') + 1):
        labels = np.append(labels, chr(l))

    for l1 in range(ord('A'), ord('Z') + 1):
        for l2 in range(ord('A'), ord('Z') + 1):
            labels = np.append(labels, chr(l1) + chr(l2))

    return labels

def pre_process_y(y, reverse = False):
    shift = 1500
    
    if reverse:
        return np.exp(y) - shift
    else:
        return np.log(y + shift)

def pre_process(X, y):
    labels = get_AZ_labels()
    
    encoder = LabelEncoder()
    encoder.fit(labels)

    split = len(X.columns[X.columns.map(lambda column: column.startswith('cat'))])
    for i in range(0, split):
        col = X.columns[i]

        X[col] = encoder.transform(X[col])

    for i in range(split, X.shape[1]):
        col = X.columns[i]

        X[col] = boxcox(X[col] + 1)[0]

    return X.values, pre_process_y(y.values)

def fit(X, y):
    clf = XGBRegressor(n_estimators = 1000, max_depth = 3, reg_alpha = 1.3)
    clf.fit(X, y)
    return clf
