import csv
import numpy as np
import pandas as pd


path  = r"C:\Users\ASUS\Desktop\machinelearning2-labwork\winequality\winequality-red.csv"

data = pd.read_csv(path,sep =';', header = 0)
"""print(data.head())
print(data.shape)"""
X =  data.iloc[:,:-1].values

"""print(X.shape) #(1599, 11)"""

Y = data.iloc[:,11].values

"""print(Y)
print(type(Y)) #<class 'numpy.ndarray'>

print(Y.shape) #(1599,)
print(Y.ndim)
"""
Y = Y.reshape(1599,1)
"""print('new shape of Y')
print(Y.shape)"""

def calculate_mean(X):
    return np.mean(X,axis = 0)

def caculate_varience(X):
    return np.var(X,axis = 0)

def calculate_cov():
    #initialize the list to append the error between a value and the mean
    mat_x =[]
    mat_y =[]
    #find the error between each x and mean x
    for x in X:
        x_group = x - mean_value_X
        mat_x.append(x_group)
    #print(len(mat_x))
    #find the error between each y and mean y
    for y in Y:
        y_group = y - mean_value_Y
        mat_y.append(y_group)
    #print(len(mat_y))
    #the multiplication of each (x-meanx) and each(y - meany)
    numerator_temp = [mat_x[i] * mat_y[i] for i in range(len(mat_y))]
    #numerator of cov
    numerator = sum(numerator_temp)

    return numerator/len(numerator_temp)

def calculate_correalation():
    numerator_vector = []
    for x in std_x:
        numerator = x * std_y
        numerator_vector.append(numerator)
    
    numerator_vector = np.asarray(numerator_vector)
    print('----')
    print(numerator_vector.shape)
    correalation = np.divide(covarience_value,numerator_vector)
    return correalation
    
    
mean_value_X = calculate_mean(X)
print(mean_value_X)
print(mean_value_X.shape) #1599,
mean_value_Y = calculate_mean(Y)
varience_value = caculate_varience(X)
covarience_value = calculate_cov()
print(varience_value.shape)


#print(covarience_value.shape) #11,


std_x = np.std(X,axis = 0)
#print(std_x)
print(std_x.shape) 
std_y = np.std(Y,axis = 0)
#print(std_y)
#print(std_y.shape) #1,

numerator = calculate_correalation()
print(type(numerator))
print(numerator.shape)
print(numerator)


"""print(mean_value)
print(varience_value)"""



    
    


    


















