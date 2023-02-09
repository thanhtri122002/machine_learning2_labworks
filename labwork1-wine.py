import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

path = r"C:\Users\ASUS\Desktop\machinelearning2-labwork\winequality-red.csv"

data = pd.read_csv(path, sep=';', header=0)

column_names = list(data.columns.values.tolist())

X = [column_name for column_name in column_names if column_name != "quality"]
Y = ["quality"]

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

#correalation_features = X_data.corr()


print("correlation")

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


print("PCA components:\n", X_pca)
print("The total variation in the data is explained by the first two principal components is:",
      explained_variance[0] + explained_variance[1])

plt.figure(figsize=(10, 10))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data['quality'], cmap='plasma')
plt.xlabel('The first principal component')
plt.ylabel('The second principal component')

# Adding a title to the chart
plt.title("2D Distribution of the Data after PCA")
# Adding a color legend to the chart
cbar = plt.colorbar()
cbar.set_label("Quality")
plt.show()
