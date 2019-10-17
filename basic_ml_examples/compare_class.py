from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate
from data_fetch import get_data



def get_clf_name(clf): return str(clf).split("(")[0]

data_sets = ["Breast Cancer",
                 "Blood Transfusion",
                 "Diabetes",
                 "Credit Scores",
                 "Oil Spill"]

#####################        Set varibles              ###########################

dataset = data_sets[0]              # pick data set
FOLDS = 5                           # set # of folds for cross validation

# clfs = [RandomForestClassifier(),  # *************
#         AdaBoostClassifier()]      # now list classifiers

#
# clfs = [RandomForestClassifier(max_depth=1),
#         RandomForestClassifier(max_depth=2),
#         RandomForestClassifier(max_depth=4),
#         RandomForestClassifier(max_depth=8), ]


####################          Get data                  ###########################

features, labels = get_data(dataset)


###################          Fit classifier            ###########################

# results = {}
# for clf in clfs:                  # ***************** add loop over classifiers
#     result = cross_validate(clf, features, labels, cv=FOLDS)
#     results[get_clf_name(clf)] = result

params = {"n_estimators":[10, 100], "max_depth": [1,3,5]}

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(RandomForestClassifier(), params)
clf.fit(features, labels)

print(sorted(clf.cv_results_.keys()))


###################          Display result          ###########################
#
# for name, result in results.items(): # ***************** add loop over results
#     print("Cv results for", name)
#     for res in result["test_score"]: print(res)
#
# from matplotlib import pyplot as plt
# import numpy as np
#
# fig = plt.figure()
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
#
# all_scores = np.transpose([np.array(results[key]["test_score"]) for key in results])
# ax.boxplot(all_scores)
#
# ax.set_xticks([1,2])
# ax.set_xticklabels([get_clf_name(clf) for clf in clfs])
# ax.set_yticks([0.8,1])
# plt.show()
