import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans  
from sklearn.metrics import davies_bouldin_score
#take the path of the datasets
path_1 = r'C:\Users\ASUS\Desktop\machinelearning2-labwork\labwork2\Raisin_Dataset.xlsx'
path_2 = r"C:\Users\ASUS\Desktop\machinelearning2-labwork\labwork2\abalone.csv"

#dataset 1 
dataset_1 = pd.read_excel(path_1,header = 0)
columns_1 = list(dataset_1.columns.values.tolist())
Class_mapping ={'Kecimen':0, 'Besni':1}
dataset_1["Class"] = dataset_1["Class"].map(Class_mapping)
column_names_1 = [column_name for column_name in columns_1 if column_name != "Class"]

X_data_1 = dataset_1.loc[:,column_names_1].values
Y_data_1 = dataset_1.loc[:,["Class"]]
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
Y_data_2 = dataset_2.loc[:,["Rings"]]
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

def PCA_visualize(dataset,target_values,target_name):
    dataset = StandardScaler().fit_transform(dataset)
    pca = PCA(n_components = 5)
    X_pca = pca.fit_transform(dataset)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=target_values[target_name], cmap='viridis')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar()
    plt.title('Data Projected onto First Two Principal Components')
    plt.show()
    return X_pca
    


def training_kmeans(dataset):
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state= 42)  
    y_predict= kmeans.fit_predict(dataset)
    score = davies_bouldin_score(dataset,y_predict)
    plt.scatter(dataset[y_predict == 0, 0], dataset[y_predict == 0, 1], s = 100, c = 'red', label = 'Cluster 1') # plotting cluster 2
    plt.scatter(dataset[y_predict == 1, 0], dataset[y_predict == 1, 1], s = 100, c = 'blue', label = 'Cluster 2') # plotting cluster 3
    plt.scatter(dataset[y_predict == 2, 0], dataset[y_predict == 2, 1], s = 100, c = 'green', label = 'Cluster 3') # plotting cluster 4
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroid')
    # plot title addition
    plt.title('Clusters ')
    # labelling the x-axis
    
    # label of the y-axis

    # printing the legend
    plt.legend()
    # show the plot
    plt.show()
    return score

def training_kmeans_pca(pca):
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state= 42)  
    y_predict= kmeans.fit_predict(pca)
    plt.scatter(pca[y_predict == 0, 0], pca[y_predict == 0, 1], s = 100, c = 'red', label = 'Cluster 1') # plotting cluster 2
    plt.scatter(pca[y_predict == 1, 0], pca[y_predict == 1, 1], s = 100, c = 'blue', label = 'Cluster 2') # plotting cluster 3
    plt.scatter(pca[y_predict == 2, 0], pca[y_predict == 2, 1], s = 100, c = 'green', label = 'Cluster 3') # plotting cluster 4
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroid')
    # plot title addition
    plt.title('Clusters with pca')
    # labelling the x-axis
    # label of the y-axis
    # printing the legend
    plt.legend()
    # show the plot
    plt.show()


pca1 = PCA_visualize(X_data_1,Y_data_1,"Class")
pca2= PCA_visualize(X_data_2,Y_data_2,"Rings")
score1 = training_kmeans(X_data_1)
score2 = training_kmeans(X_data_2)
training_kmeans_pca(pca1)
training_kmeans_pca(pca2)

print(score1)
print(score2)





  





