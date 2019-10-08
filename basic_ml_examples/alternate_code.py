# #################       Alternate Fit classifer   #################################
# from sklearn.model_selection import StratifiedKFold
# result = []
# skf = StratifiedKFold(n_splits=FOLDS, shuffle=True)
#
# for train_index, test_index in skf.split(features, labels):
#     X_train, X_test = features[train_index], features[test_index]
#     y_train, y_test = labels[train_index], labels[test_index]
#
#     clf.fit(X_train, y_train)
#     result.append(clf.score(X=X_test, y=y_test))
# result = {"test_score":result}      # makes sure it works with display result code





# ###################          Alternate display result          ###########################
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
# ax.set_yticks([0.5,1])
# plt.show()