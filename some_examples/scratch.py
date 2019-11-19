import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
from sklearn import datasets
from pprint import pprint as pp


from sklearn.datasets import load_boston

data = load_boston()

X = data["data"]
y = data["target"]

plt.scatter(X[:,0],y)

plt.scatter(X[:,0], X[:,6], c=y)



from sklearn.preprocessing import scale
X = scale(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train, y_train)



