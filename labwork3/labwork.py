import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler , normalize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split ,LeaveOneOut, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


def take_dataset():
    path_1 = r'./labwork2/Raisin_Dataset.xlsx'
    path_2 = r'./labwork2/abalone.csv'
    path_3 = r'./labwork1/iris.csv'
    while True:
        print('1. Abalone dataset')
        print('2. Raisin dataset')
        print('3. Iris dataset')
        print('choose a choice')
        choice = int(input('select choice: '))
        if choice == 1:
            dataset_1 = pd.read_excel(path_1,header = 0)
            columns_1 = list(dataset_1.columns.values.tolist())
            Class_mapping ={'Kecimen':0, 'Besni':1}
            dataset_1["Class"] = dataset_1["Class"].map(Class_mapping)
            print('Raisin')
            print(dataset_1.head())
            column_names_1 = [column_name for column_name in columns_1 if column_name != "Class"]
            X_data_1 = dataset_1.loc[:,column_names_1].values
            Y_data_1 = dataset_1.loc[:,["Class"]]
            return X_data_1, Y_data_1

        elif choice == 2:
            dataset_2 = pd.read_csv(path_2,header = None)
            dataset_2.columns = ["Sex","Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]
            Sex_mapping = {'M': 0 , "F": 1, "I": 2}
            dataset_2["Sex"] = dataset_2["Sex"].map(Sex_mapping)
            print('Raisin ')
            print(dataset_2.head())
            columns_2 = list(dataset_2.columns.values.tolist())
            column_names_2 = [column_name for column_name in columns_2 if column_name != "Rings" and column_name !="Sex"]
            X_data_2 = dataset_2.loc[:,column_names_2].values
            Y_data_2 = dataset_2.loc[:,["Rings"]]
            return X_data_2, Y_data_2

        elif choice == 3:
            dataset_3 = pd.read_csv(path_3)
            species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, "Iris-virginica": 2}
            dataset_3['species']  = dataset_3['species'].map(species_mapping)
            print('Iris')
            print(dataset_3.head())
            columns_name = list(dataset_3.columns.values.tolist())
            X = [column_name for column_name in columns_name if column_name != "species"]
            Y = ["species"]
            
            X_data_3 = dataset_3.loc[:, X].values
            Y_data_3 = dataset_3.loc[:, Y].values
            return X_data_3, Y_data_3
        
            
            
def normalize_data(X_data):
    return normalize(X_data)
    
def app_PCA(X_data,components):
    pca = PCA(n_components=components)
    x_pca= pca.fit_transform(X_data)
    return x_pca

def app_SVD(X_data, components):
    svd = TruncatedSVD(n_components= components)
    X_trans = svd.fit_transform(X_data)
    return X_trans

def app_loo(X_data,Y_data, no_neighbors):
    classifier = KNeighborsClassifier(n_neighbors= no_neighbors)
    loo = LeaveOneOut()
    scores = cross_val_score(classifier, X_data, Y_data, cv = loo)
    return scores

def cross_val(X_data,Y_data, no_neighbors):
    
    classifier = KNeighborsClassifier(n_neighbors=no_neighbors)
    k_folds = KFold(n_splits = 5)
    scores = cross_val_score(classifier, X_data, Y_data, cv=k_folds)
    return scores

def train(X_data,Y_data, no_neighbors):
    x_train , x_test ,y_train , y_test = train_test_split(X_data, Y_data, test_size= 0.25,random_state= 0)
    st_x = StandardScaler()
    x_train = st_x.fit_transform(x_train)
    x_test = st_x.fit_transform(x_test)
    print(x_train)
    classifier = KNeighborsClassifier(n_neighbors= no_neighbors)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test) #type : numpy array
    
    cm = confusion_matrix(y_test,y_pred) #type : numpy array
    
    print(cm)

def perceptron_classifier(X_data,Y_data):
    x_train , x_test ,y_train , y_test = train_test_split(X_data, Y_data, test_size= 0.25,random_state= 0)
    p = Perceptron()
    return p.fit(x_train,y_train) , x_test , y_test

def plot_decision_boundary(model, X, Y, cmap='Paired_r'):
    h = 0.02
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.array[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k')
    plt.show()


if __name__ == "__main__":
    X_data, Y_data = take_dataset()
    while True:
        print('1. work with raw data')
        print('2. work with normalize data')
        print('3. work with pca')
        print('4. work with svd')
        print('5. perceptron visualize')
        choice = int(input('select choice: '))
        if choice == 1:
            normalize_data(X_data)
            neighbors = int(input('select the number of neighbors: '))
            train(X_data,Y_data,neighbors)
        elif choice == 2:
            neighbors = int(input('select the number of neighbors: '))
            train(X_data,Y_data,neighbors)
            print('----')
        elif choice == 3:
            neighbors = int(input('select the number of neighbors: '))
            components = int(input('select no components: '))
            x_pca = app_PCA(X_data,components)
            train(X_data,Y_data,neighbors)
        elif choice == 4:
            neighbors = int(input('select the number of neighbors: '))
            components = int(input('select no components: '))
            x_svd = app_SVD(X_data,components)
            print(x_svd.shape)
            train(X_data, Y_data, neighbors)
        elif choice ==5:
            neighbors = int(input('select the number of neighbors: '))
            components = int(input('select no components: '))
            x_svd = app_SVD(X_data, components)
            model,x_test,y_test = perceptron_classifier(x_svd,Y_data)
            visualize = plot_decision_boundary(model,x_svd,y_test)

        else:
            break
        

