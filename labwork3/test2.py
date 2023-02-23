from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut


# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Initialize the KNN model with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model on the training set
knn.fit(X_train, y_train)

# Predict the class labels for the test set
y_pred = knn.predict(X_test)

# Calculate the classification error
error = 1 - accuracy_score(y_test, y_pred)
print(f"Classification error: {error:.2f}")


# Initialize the KNN model with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model on the training set
knn.fit(X_train, y_train)

# Predict the class labels for the test set
y_pred = knn.predict(X_test)

# Calculate the classification error
error = 1 - accuracy_score(y_test, y_pred)
print(f"Classification error (k=5): {error:.2f}")

# Initialize the KNN model with k=5
knn = KNeighborsClassifier(n_neighbors=40)

# Train the model on the training set
knn.fit(X_train, y_train)

# Predict the class labels for the test set
y_pred = knn.predict(X_test)

# Calculate the classification error
error = 1 - accuracy_score(y_test, y_pred)
print(f"Classification error (k=40): {error:.2f}")

def PCA_SVD():
    # Apply PCA on the data
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Apply SVD on the data
    svd = TruncatedSVD(n_components=2)
    X_train_svd = svd.fit_transform(X_train)
    X_test_svd = svd.transform(X_test)

    # Initialize the KNN model with k=3
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the model on the PCA-transformed training set
    knn.fit(X_train_pca, y_train)

    # Predict the class labels for the PCA-transformed test set
    y_pred_pca = knn.predict(X_test_pca)

    # Calculate the classification error for the PCA-transformed data
    error_pca = 1 - accuracy_score(y_test, y_pred_pca)
    print(f"Classification error (PCA): {error_pca:.2f}")

    # Train the model on the SVD-transformed training set
    knn.fit(X_train_svd, y_train)

    # Predict the class labels for the SVD-transformed test set
    y_pred_svd = knn.predict(X_test_svd)

    # Calculate the classification error for the SVD-transformed data
    error_svd = 1 - accuracy_score(y_test, y_pred_svd)
    print(f"Classification error (SVD): {error_svd:.2f}")
    
PCA_SVD()

def improve_perf():
    # Initialize the KNN model with k=3
    knn = KNeighborsClassifier(n_neighbors=3)

    # Perform 5-fold cross-validation on the original data
    cv_scores = cross_val_score(knn, iris.data, iris.target, cv=5)

    # Calculate the mean and standard deviation of the cross-validation scores
    mean_cv_score = cv_scores.mean()
    std_cv_score = cv_scores.std()

    print(f"Mean cross-validation score: {mean_cv_score:.2f}")
    print(f"Standard deviation of cross-validation scores: {std_cv_score:.2f}")

improve_perf()

def leave_one_out():
    # Initialize the KNN model with k=3
    knn = KNeighborsClassifier(n_neighbors=3)

    # Apply leave-one-out cross-validation
    loo = LeaveOneOut()
    errors = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        errors.append(y_pred != y_test)

    # Calculate the mean classification error
    error_loo = sum(errors) / len(errors)
    print("Classification error (leave-one-out):", error_loo)
    
leave_one_out()