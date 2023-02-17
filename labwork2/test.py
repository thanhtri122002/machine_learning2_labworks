import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


X = np.array([1, 2, 9, 12, 20])
print(X.shape)
X = X.reshape((X.shape[0],1))

single_linkage = linkage(X, method = 'single')
dendrogram_single = dendrogram(single_linkage)
plt.title("Single Linkage")
plt.show()

complete_linkage = linkage(X, method="complete")
dendrogram_complete = dendrogram(complete_linkage)
plt.title("Complete Linkage")
plt.show()

path_1 = r'C:\Users\ASUS\Desktop\machinelearning2-labwork\labwork1\winequality-red.csv'
path_2 = r'C:\Users\ASUS\Desktop\machinelearning2-labwork\labwork1\iris.csv'
dataset_1 = pd.read_csv(path_1,header = 0, sep = ";", )
print(dataset_1.head())

X = dataset_1.loc[:,:]

dataset_2 = pd.read_csv(path_2, header = None)









