import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

path_2  = r"C:\Users\ASUS\Desktop\machinelearning2-labwork\iris.txt"

data_iris = pd.read_csv(path_2,sep =',', header = None )

data_iris.columns = ['sepal length', 'sepal width',
                     'petal length', 'petal width', 'species']
print(data_iris.head())
print(type(data_iris))


column_names = list(data_iris.columns.values.tolist())
X = [column_name for column_name in column_names if column_name !='species']

Y = ['species']

X_data = data_iris.loc[:,X]


encoded = pd.get_dummies(data= data_iris, columns= ["species"] , dummy_na= False)

print(encoded)
Y_data = encoded.loc[:,["species_Iris-setosa","species_Iris-versicolor","species_Iris-virginica"]]

print(X_data)
print(Y_data)

X_mean = X_data.mean()
Y_mean = Y_data.mean()

print(X_mean)
print(Y_mean)

X_varience = data_iris.var()
print(X_varience)

covarience = data_iris.cov()
print(covarience)

correalation = data_iris.corr()
correalation_features = X_data.corr()

print("correalation")

print(correalation)

print("correalation between features")

print(correalation_features)


    



