import csv
import numpy as np
import pandas as pd

path  = r"C:\Users\ASUS\Desktop\machinelearning2-labwork\winequality\winequality-red.csv"

data = pd.read_csv(path,sep =';', header = 0)

print(data.head())
column_names = list(data.columns.values.tolist())

print(column_names)
print(type(column_names))
 
X = [column_name for column_name in column_names if column_name !="quality"]
Y = ["quality"]
"""print("attribute X")
print(X)
print(type(X))
print(Y)
print(type(Y))"""

X_data = data.loc[:, X]

Y_data = data.loc[:, Y]

print(type(X_data))

print(type(Y_data))

print(X_data)
print(Y_data)

#calculate the mean 

X_mean = X_data.mean()
Y_mean = Y_data.mean()

print(X_mean)
print(Y_mean)

X_varience = X_data.var()


print(X_varience)

covarience = data.cov()

print("The covarience")

print(covarience)

correalation = data.corr()
correalation_features = X_data.corr()


print("correalation")

print(correalation)

print("correalation between features")

print(correalation_features)









