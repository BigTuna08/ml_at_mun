from practice import loader
from sklearn.metrics import normalized_mutual_info_score


def score_clf(clf=None):
    if clf:
        Xt = loader("X_test")
        yt = loader("y_test")
        print("Classifier accuracy:", clf.score(Xt, yt))
    else:
        print("No classifier given")


def score_clusters_train(assignmnet=None):
    if assignmnet:
        true_cl = loader("cl")
        print("Classifier accuracy:", normalized_mutual_info_score(assignmnet, true_cl))
    else:
        print("No training clustering given")


def score_clusters_test(assignmnet=None):
    if assignmnet:
        true_cl = loader("cl_test")
        print("Classifier accuracy:", normalized_mutual_info_score(assignmnet, true_cl))
    else:
        print("No training clustering given")

