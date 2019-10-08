from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from data_fetch import get_data


data_sets = ["Breast Cancer",
                 "Blood Transfusion",
                 "Diabetes",
                 "Credit Scores",
                 "Oil Spill"]

#####################        Set varibles              ###########################

dataset = data_sets[0]              # pick data set
FOLDS = 5                           # set # of folds for cross validation
clf = RandomForestClassifier()      # choose classifier



####################          Get data                  ###########################

features, labels = get_data(dataset)


###################          Fit classifier            ###########################

result = cross_validate(clf, features, labels, cv=FOLDS)



###################          Display result          ###########################

print("Cv results")
for res in result["test_score"]: print(res)
