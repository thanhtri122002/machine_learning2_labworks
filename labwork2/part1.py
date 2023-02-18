import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

#Apply the AHC for the 1-D array
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

#take the path of datasets
path_1 = r'C:\Users\ASUS\Desktop\machinelearning2-labwork\labwork2\Raisin_Dataset.xlsx'
path_2 = r'C:\Users\ASUS\Desktop\machinelearning2-labwork\labwork2\abalone.csv'

#Study the data set 1
dataset_1 = pd.read_excel(path_1,header=0 )
columns_1 = list(dataset_1.columns.values.tolist())
X_1 = [column_name for column_name in columns_1 if column_name !="Class"]
X_data_1 = dataset_1.loc[:,X_1].values
mean_1 = dataset_1.mean()
varience_1 = dataset_1.var()
covarience_1 = dataset_1.cov()
corealation_1 = dataset_1.corr()
corr_max_1 = corealation_1.abs().unstack().sort_values(ascending=False)
corr_max_1 = corr_max_1[corr_max_1 != 1]
most_highly_corealated_1 = corr_max_1.index[0]
print('the mean of the dataset 1')
print(mean_1)
print('the varience of the dataset 1: ')
print(varience_1)
print('the covarience of the dataset 1: ')
print(covarience_1)
print('the corealation of dataset 1: ')
print(corealation_1)
print('the most highly corealated in dataset 1:')
print(most_highly_corealated_1)
single_linkage_1 = linkage(X_data_1, method = "complete")
dendrogram_single_1 = dendrogram(single_linkage_1)
plt.show()

# Study the dataset 2
dataset_2 = pd.read_csv(path_2)
#adding names to the columns 
dataset_2.columns =["Sex","Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]
print(dataset_2.head())
#preparing for the label encoding 
Sex_mapping = {'M': 0 , "F": 1, "I": 2}
dataset_2["Sex"] =dataset_2["Sex"].map(Sex_mapping)
columns_2 = list(dataset_2.columns.values.tolist())
X_2 = [column_name for column_name in columns_2 if column_name != "Rings"]
X_data_2 = dataset_2.loc[:,X_2].values
mean_2 = dataset_2.mean()
varience_2 = dataset_2.var()
covarience_2 = dataset_2.cov()
corealation_2 = dataset_2.corr()
corr_max_2 = corealation_2.abs().unstack().sort_values(ascending=False)
corr_max_2 = corr_max_2[corr_max_2 != 1]
most_highly_corealated_2 = corr_max_2.index[0]
print('mean of the dataset 2: ')
print(mean_2)
print("varience of the dataset 2: ")
print(varience_2)
print('covarience of the dataset 2: ')
print(covarience_2)
print('corealation of the dataset 2: ')
print(corealation_2)
print('the most highly corealated in dataset 2: ')
print(most_highly_corealated_2)
single_linkage_2 = linkage(X_data_2, method = "complete")
dendogram_singe_2 = dendrogram(single_linkage_2)
plt.show()









