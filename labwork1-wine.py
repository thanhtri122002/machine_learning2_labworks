import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

path_1  = r"C:\Users\ASUS\Desktop\machinelearning2-labwork\winequality\winequality-red.csv"

data_wine = pd.read_csv(path_1,sep =';', header = 0)





print(data_wine.head())
column_names = list(data_wine.columns.values.tolist())

print(column_names)
print(type(column_names))
 
X = [column_name for column_name in column_names if column_name !="quality"]
Y = ["quality"]

X_data = data_wine.loc[:, X]

Y_data = data_wine.loc[:, Y]

print(type(X_data))

print(type(Y_data))

print(X_data)
print(Y_data)

#calculate the mean 
def calculate_mean(data):
    return data.mean()

def calculate_var(data):
    return data.var()

def calculate_cov(data):
    return data.cov()

def calculate_corr(data):
    return data.corr()


X_mean = X_data.mean()
Y_mean = Y_data.mean()

print(X_mean)
print(Y_mean)

X_varience = X_data.var()


print(X_varience)

covarience = data_wine.cov()

print("The covarience")

print(covarience)

correalation = data_wine.corr()
correalation_features = X_data.corr()


print("correalation")

print(correalation)

print("correalation between features")

print(correalation_features)
