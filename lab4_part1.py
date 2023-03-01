import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, KFold , cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.datasets import load_iris

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import accuracy_score

def standarlization(x_train , x_test):
    st = StandardScaler()
    x_train_standardlized = st.fit_transform(x_train)
    x_test_standardlized = st.fit_transform(x_test)
    return x_train_standardlized , x_test_standardlized
def iris_data():
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    iris_df = iris_df.replace({'species': {0: 'setosa', 1: 'versicolor', 2: 'virginica'}})
    X = iris_df.iloc[:, :-1]
    y = iris_df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)
    x_train_st, x_test_st = standarlization(x_train, x_test)
    return x_train_st, x_test_st, y_train, y_test

def decision_tree():
    x_train_st , x_test_st, y_train , y_test = iris_data()
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train_st, y_train)
    y_pred = clf.predict(x_test_st)
    # Calculate the accuracy of the classification
    accuracy = accuracy_score(y_test, y_pred)
    error = 1 - accuracy
    # Print the results
    print("Accuracy:", accuracy)
    print("Error:", error)
print("Decision Tree")  
decision_tree()

