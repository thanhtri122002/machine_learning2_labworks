import pandas as pd
from sklearn.model_selection import KFold 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# Load the dataset

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(url, sep=';')

# Split the data into features and labels
X = data.drop(['quality'], axis=1)
y = data['quality']

# Create 100 training sets and 1 testing set
n_splits = 100
test_size = 0.2
random_state = 42
kfold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

X_train_list = []
y_train_list = []
for i, (train_index, test_index) in enumerate(kfold.split(X)):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_train_list.append(X_train)
    y_train_list.append(y_train)
X_test, y_test = X.iloc[test_index], y.iloc[test_index]

# Train 100 decision trees using the bagging technique
n_estimators = 100
model = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=n_estimators)
error_one_list = []
error_all_list = []
for X_train, y_train in zip(X_train_list, y_train_list):
    model.fit(X_train, y_train)

    # Calculate the error of classification using one decision tree
    y_pred_one = model.estimators_[0].predict(X_test)
    error_one = sum(y_test != y_pred_one) / len(y_test)
    error_one_list.append(error_one)

    # Calculate the error of classification using all decision trees
    y_pred_all = model.predict(X_test)
    error_all = sum(y_test != y_pred_all) / len(y_test)
    error_all_list.append(error_all)

avg_error_one = sum(error_one_list) / len(error_one_list)
avg_error_all = sum(error_all_list) / len(error_all_list)

print('Average error of classification using one decision tree: ', avg_error_one)
print('Average error of classification using all decision trees: ', avg_error_all)
