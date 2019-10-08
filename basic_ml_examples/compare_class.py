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

clfs = [RandomForestClassifier(),  # *************
        AdaBoostClassifier()]      # now list classifiers



####################          Get data                  ###########################

features, labels = get_data(dataset)


###################          Fit classifier            ###########################

results = {}
for clf in clfs:                  # ***************** add loop over classifiers
    result = cross_validate(clf, features, labels, cv=FOLDS)
    results[get_clf_name(clf)] = result



###################          Display result          ###########################

for name, result in results.items(): # ***************** add loop over results
    print("Cv results for", name)
    for res in result["test_score"]: print(res)
