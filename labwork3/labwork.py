import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron


def take_dataset():
    path_1 = r'./labwork2/Raisin_Dataset.xlsx'
    path_2 = r'./labwork2/abalone.csv'
    path_3 = r'./labwork1/iris.csv'
    while True:
        print('1. Raisin dataset')
        print('2. Abalone dataset')
        print('3. Iris dataset')
        print('choose a choice')
        choice = int(input('select choice: '))
        if choice == 1:
            dataset_1 = pd.read_excel(path_1, header=0)
            columns_1 = list(dataset_1.columns.values.tolist())
            Class_mapping = {'Kecimen': 0, 'Besni': 1}
            dataset_1["Class"] = dataset_1["Class"].map(Class_mapping)
            print('Raisin')
            print(dataset_1.head())
            column_names_1 = [column_name for column_name in columns_1 if column_name != "Class"]
            X_data = dataset_1.loc[:, column_names_1].values
            #Y_data = dataset_1.loc[:,["Class"]]
            Y_data = dataset_1["Class"].values
            flag = 1
            return X_data, Y_data, flag

        elif choice == 2:
            dataset_2 = pd.read_csv(path_2, header=None)
            dataset_2.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight",
                                 "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
            Sex_mapping = {'M': 0, "F": 1, "I": 2}
            dataset_2["Sex"] = dataset_2["Sex"].map(Sex_mapping)
            print('Raisin ')
            print(dataset_2.head())
            columns_2 = list(dataset_2.columns.values.tolist())
            column_names_2 = [column_name for column_name in columns_2 if column_name != "Rings" and column_name != "Sex"]
            X_data = dataset_2.loc[:, column_names_2].values
            #Y_data = dataset_2.loc[:,["Rings"]]
            Y_data = dataset_2["Rings"].values
            flag = 2
            return X_data, Y_data, flag
        if choice == 3:
            dataset_3 = pd.read_csv(path_3)
            species_mapping = {'Iris-setosa': 0,'Iris-versicolor': 1, "Iris-virginica": 2}
            dataset_3['species'] = dataset_3['species'].map(species_mapping)
            print('Iris')
            print(dataset_3.head())
            columns_name = list(dataset_3.columns.values.tolist())
            X = [column_name for column_name in columns_name if column_name != "species"]
            Y = ["species"]

            X_data = dataset_3.loc[:, X].values
            Y_data = dataset_3.loc[:, Y].values
            flag = 3
            return X_data, Y_data, flag


def normalize_data(X_data):
    return normalize(X_data)


def app_PCA(X_data, components):
    pca = PCA(n_components=components)
    x_pca = pca.fit_transform(X_data)
    return x_pca

def app_SVD(X_data, components):
    svd = TruncatedSVD(n_components=components)
    X_trans = svd.fit_transform(X_data)
    return X_trans


def app_loo(X_data, Y_data, no_neighbors):
    knn = KNeighborsClassifier(n_neighbors= no_neighbors)

    # Apply leave-one-out cross-validation
    loo = LeaveOneOut()
    errors = []

    for train_index, test_index in loo.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = Y_data[train_index], Y_data[test_index]
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        errors.append(y_pred != y_test)

    # Calculate the mean classification error
    error_loo = sum(errors) / len(errors)
    print("Classification error (leave-one-out):", error_loo)


def cross_val(X_data, Y_data, no_neighbors):
    classifier = KNeighborsClassifier(n_neighbors=no_neighbors)
    k_folds = KFold(n_splits=5)
    scores = cross_val_score(classifier, X_data, Y_data, cv=k_folds)
    return scores

def train(X_data, Y_data, no_neighbors):
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
    classifier = KNeighborsClassifier(n_neighbors=no_neighbors)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)  # type : numpy array
    cm = confusion_matrix(y_test, y_pred)  # type : numpy array
    precision = precision_score(y_test, y_pred, average= None)
    f1score = f1_score(y_test, y_pred, average= None)
    accuracy = accuracy_score(y_test, y_pred )
    recall = recall_score(y_test, y_pred, average = None)
    error = 1 - accuracy
    print(f'confusion matrix : {cm}')
    print(f'f1 score : {f1score}')
    print(f'accuracy : {accuracy}')
    print(f'recall :{recall}')
    print(f'classification error : {error}')


def perceptron_classifier(X_data, Y_data):
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.25, random_state=0)
    p = Perceptron()
    return p.fit(x_train, y_train), x_test, y_test


def plot_linear_classifier(X, y, title, components):
    # Apply PCA to reduce dimensions to 2D
    pca = PCA(n_components=components)
    X_reduced = pca.fit_transform(X)

    # Standardize data for Perceptron classifier
    scaler = StandardScaler()
    X_reduced = scaler.fit_transform(X_reduced)

    # Train a Perceptron classifier
    clf = Perceptron()
    clf.fit(X_reduced, y)

    # Create meshgrid to plot decision boundary
    h = 0.05
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #y_numeric = y['Rings'].astype(int)

    # Plot the decision boundary and data points
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X_reduced[:, 0], X_reduced[:,1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    X_data, Y_data, flag=take_dataset()
    while True:
        print('1. work with raw data')
        print('2. work with normalize data')
        print('3. work with pca')
        print('4, work with svd')
        print('5. perceptron visualize')
        print('6 .Leave one out ')
        print('Or press any key to exit')
        choice=int(input('select choice: '))
        if choice == 1:
            neighbors=int(input('select the number of neighbors: '))
            train(X_data, Y_data, neighbors)
        elif choice == 2:
            X_data = normalize_data(X_data)
            neighbors=int(input('select the number of neighbors: '))
            train(X_data, Y_data, neighbors)
            print('----')
        elif choice == 3:
            neighbors=int(input('select the number of neighbors: '))
            components=int(input('select no components: '))
            x_pca=app_PCA(X_data, components)
            train(X_data, Y_data, neighbors)
        elif choice == 4:
            neighbors=int(input('select the number of neighbors: '))
            components=int(input('select no components: '))
            x_pca=app_SVD(X_data, components)
            train(x_pca, Y_data, neighbors)
        elif choice == 5:
            #neighbors = int(input('select the number of neighbors: '))
            components=2
            if flag == 1:
                components=2
                plot_linear_classifier(X_data, Y_data, 'Raisin dataset', components)
            elif flag == 2:
                components=2
                plot_linear_classifier(X_data, Y_data, 'Abalone dataset', components)
            elif flag == 3:
                components=2
                plot_linear_classifier(X_data, Y_data, 'Iris dataset', components)
                #x_svd = app_SVD(X_data, components)
                #model,x_test,y_test = perceptron_classifier(x_svd,Y_data)
                #visualize = plot_decision_boundary(model,x_svd,y_test)
        elif choice == 6:
            neighbors=int(input('select the number of neighbors: '))
            scores = app_loo(X_data, Y_data, neighbors)
            print(scores)
        else:
            break
