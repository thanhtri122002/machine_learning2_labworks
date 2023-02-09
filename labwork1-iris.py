import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

path = r"C:\Users\ASUS\Desktop\machinelearning2-labwork\iris.csv"
data = pd.read_csv("iris.csv")
print(data.head())
species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, "Iris-virginica": 2}


data['species'] = data['species'].map(species_mapping)


column_names = list(data.columns.values.tolist())

X = [column_name for column_name in column_names if column_name != "species"]
Y = ["species"]

X_data = data.loc[:, X].values

Y_data = data.loc[:, Y].values

#calculate the mean

mean = data.mean()

print("The mean of the data is :\n", mean)

varience = data.var()


print("The varience of the data is:\n", varience)

covarience = data.cov()

print("The covarience")

print(covarience)

corr = data.corr()

print("correalation")

print(corr)

corr_max = corr.abs().unstack().sort_values(ascending=False)

# Exclude the diagonal values
corr_max = corr_max[corr_max != 1]

# Get the most highly correlated couple of features
most_highly_correlated = corr_max.index[0]

# Print the result
print("The most highly correlated couple of features are:", most_highly_correlated)

# Standardize the data
X_data = StandardScaler().fit_transform(X_data)

pca = PCA(n_components=2)

# Fit and transform the data
X_pca = pca.fit_transform(X_data)

# Calculate the explained variance ratio
explained_variance = pca.explained_variance_ratio_

print("PCA:", pca)
print("PCA components:\n", X_pca)
print("The total variation in the data is explained by the first two principal components is:",
      explained_variance[0] + explained_variance[1])

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_data.ravel(), cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar().set_ticks([0, 1, 2])
plt.title('Iris Data Projected onto First Two Principal Components')
plt.show()
