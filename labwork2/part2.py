import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans  
#take the path of the datasets
path_1 = r'C:\Users\ASUS\Desktop\machinelearning2-labwork\labwork2\Raisin_Dataset.xlsx'
path_2 = r"C:\Users\ASUS\Desktop\machinelearning2-labwork\labwork2\abalone.csv"

#dataset 1 
dataset_1 = pd.read_excel(path_1,header = 0)
columns_1 = list(dataset_1.columns.values.tolist())
column_names_1 = [column_name for column_name in columns_1 if column_name != "Class"]
print(column_names_1)
X_data_1 = dataset_1.loc[:,column_names_1].values

print(X_data_1)
print(X_data_1.shape )#900,7

#dataset 2
dataset_2 = pd.read_csv(path_2,header = None)
dataset_2.columns = ["Sex","Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]
Sex_mapping = {'M': 0 , "F": 1, "I": 2}
dataset_2["Sex"] = dataset_2["Sex"].map(Sex_mapping)
columns_2 = list(dataset_2.columns.values.tolist())
column_names_2 = [column_name for column_name in columns_2 if column_name != "Rings" and column_name !="Sex"]
X_data_2 = dataset_2.loc[:,column_names_2].values
print('----')
print(X_data_2)

wcss_list1= [] 
wcss_list2= [] 
def find_K(dataset,wcss):
    for i in range(1,11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state= 42)
        kmeans.fit(dataset)
        wcss.append(kmeans.inertia_)

find_K(X_data_1,wcss_list1)
find_K(X_data_2,wcss_list2)

def plot(wcss,number):
    plt.plot(range(1, 11), wcss)  
    plt.title(f'The Elobw Method Graph{number}')  
    plt.xlabel('Number of clusters(k)')  
    plt.ylabel('wcss_list')  
    plt.show()  

plot(wcss_list1,1) #3
plot(wcss_list2,2) #3

def training_kmeans(dataset):
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state= 42)  
    y_predict= kmeans.fit_predict(dataset)


  





