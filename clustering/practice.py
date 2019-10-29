from scipy.stats import multivariate_normal, norm, expon
import numpy as np
from utils import *


def load_data():
    return loader("X"), loader("y")


def load_clusters():
    return loader("cl")




#########################           Write your code here                  ######################################


X,y = load_data()
X_test = loader("X_test")


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X,y)

from sklearn import mixture
gmm = mixture.GaussianMixture(n_components=5).fit(X)
cl = gmm.predict(X)
cl_test = gmm.predict(X_test)


#########################           Test your results here                  ######################################


score_clf(clf)                          # takes a trained classifier as input
score_clusters_train(cl)               # takes a list of integers
score_clusters_test(cl_test)                # takes a list of integers