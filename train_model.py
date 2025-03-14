import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import pandas as pd


X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser="auto")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

with open('mnist_model.pkl', 'wb') as f:
    pickle.dump(clf, f)