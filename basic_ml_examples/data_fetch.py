import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml

def convert_labels(x):
    return 1 if x == 1 else -1

convert_labels = np.vectorize(convert_labels, otypes=[np.float])



def get_data(ds_name):
    if ds_name == "Breast Cancer":
        features, labels = load_breast_cancer(return_X_y=True)
        labels = convert_labels(labels)
        return features, labels

    if ds_name == "Blood Transfusion":
        data = fetch_openml(name='blood-transfusion-service-center')
        features = data.data
        labels = np.array([1 if x == "1" else -1 for x in data.target])
        return features, labels

    if ds_name == "Diabetes":
        data = fetch_openml(name='diabetes')
        features = data.data
        labels = convert_labels(data.target)
        return features, labels

    if ds_name == "Credit Scores":
        data = fetch_openml(name='credit-g')
        features = data.data[:, :-1]
        labels = convert_labels(data.data[:, -1])
        return features, labels

    if ds_name == "Oil Spill":
        data = fetch_openml(name='oil_spill')

        features = data.data
        labels = np.array([1 if x == "1" else -1 for x in data.target])
        return features, labels
