import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model

from scipy.stats import skew
from sklearn.ensemble import IsolationForest

print("Loading...")
train_df = model.load_data()
X_df = train_df.iloc[0::, 1:-1]
y_df = train_df.iloc[0::, -1]
del train_df

print("Preprocessing...")
y_df.sort()
X, y = model.pre_process(X_df, y_df)

print("Ploting...")

plt.figure(1)

plt.subplot(211)
plt.plot(y_df.values, y, 'ro')
plt.ylabel('log(y + shift)')
plt.xlabel('y')

S = skew(X)
plt.subplot(212)
plt.plot(range(S.shape[0]), S, 'ro')
plt.ylabel('Skewness')
plt.xlabel('Features')

plt.show()
