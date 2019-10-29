
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

TIME_STR = "46_27"  # replace with time str from your data



def loader(fname):
    fstr = "data/{}_" + TIME_STR +".npy"
    return np.load(fstr.format(fname))


def score_clf(clf=None):
    if clf:
        Xt = loader("X_test")
        yt = loader("y_test")
        print("Classifier accuracy:", clf.score(Xt, yt))
    else:
        print("No classifier given")


def score_clusters_train(assignmnet=None):
    if not assignmnet is None:
        true_cl = loader("cl")
        print("Training cluster accuracy:", normalized_mutual_info_score(assignmnet, true_cl))
    else:
        print("No training clustering given")


def score_clusters_test(assignmnet=None):
    if not assignmnet is None:
        true_cl = loader("cl_test")
        print("Test cluster accuracy:", normalized_mutual_info_score(assignmnet, true_cl))
    else:
        print("No training clustering given")

