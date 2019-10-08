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




# #####################               Scale features                     ######################
# from sklearn import preprocessing
# scaler = preprocessing.StandardScaler().fit(features)
# features = scaler.transform(features)
#


# #########################             tsne                                  #######################
# from sklearn.manifold import t_sne
# from matplotlib import pyplot as plt
# tsne = t_sne.TSNE()
# fx = tsne.fit_transform(X)
# plt.scatter(fx[:,0], fx[:, 1])
# plt.show()



##########################            grid search                            #############
# params = {"n_estimators":[10, 100], "max_depth": [1,3,5]}
# from sklearn.model_selection import GridSearchCV
# clf = GridSearchCV(RandomForestClassifier())
#
# print(sorted(clf.cv_results_.keys()))

