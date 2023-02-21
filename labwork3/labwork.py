import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler , normalize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

path = r'./labwork1/iris.csv'

iris_df = pd.read_csv(path)
print(iris_df.head())
def normalize_data(X_data):
    return normalize(X_data)
    
def app_PCA(X_data):
    pca = PCA(n_components=2)
    x_pca= pca.fit_transform(X_data)
    return x_pca

def app_SVD(X_data):
    pass

def take_data():
    species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, "Iris-virginica": 2}
    iris_df['species']  = iris_df['species'].map(species_mapping)

    columns_name = list(iris_df.columns.values.tolist())

    X = [column_name for column_name in columns_name if column_name != "species"]
    Y = ["species"]
    print(X)
    X_data = iris_df.loc[:, X].values
    Y_data = iris_df.loc[:, Y].values
    return X_data, Y_data


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

if __name__ == "__main__":
    X_data, Y_data = take_data()
    while True:
        choice = int(input('select 1 to normalize the data select 2 for work with raw data: '))
        if choice == 1:
            normalize_data(X_data)
            neighbors = int(input('select the number of neighbors: '))
            train(X_data,Y_data,neighbors)
        
        elif choice == 2:
            neighbors = int(input('select the number of neighbors: '))
            train(X_data,Y_data,neighbors)
            print('----')
        else: 
            break
        

