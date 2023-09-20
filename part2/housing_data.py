import numpy as np
from sklearn.datasets import make_regression

# Generate synthetic data with sklearn
n_samples = 100  # Number of samples
n_features = 1  # Number of features (in this case, a single feature)
noise_stddev = 1.0  # Standard deviation of the noise

X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise_stddev)

# X is the feature matrix, y is the target variable

print("Feature matrix (X):")
print(X[:5])  # Print the first 5 samples of X

print("\nTarget variable (y):")
print(y[:5])  # Print the first 5 samples of y
