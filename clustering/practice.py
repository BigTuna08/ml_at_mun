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





#########################           Test your results here                  ######################################


score_clf()                          # takes a trained classifier as input
score_clusters_train()               # takes a list of integers
score_clusters_test()                # takes a list of integers