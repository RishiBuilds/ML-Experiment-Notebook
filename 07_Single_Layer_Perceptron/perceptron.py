import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("Iris.csv")

# Select only two classes for binary classification
data = data[data['Species'] != 'Iris-virginica']

# Convert labels to binary
data['Species'] = data['Species'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1
})

# Select features
X = data[['SepalLengthCm', 'SepalWidthCm']].values
y = data['Species'].values

# Initialize weights and bias
weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.01
epochs = 100

# Activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Training the perceptron
for _ in range(epochs):
    for i in range(len(X)):
        net = np.dot(X[i], weights) + bias
        y_pred = step_function(net)
        error = y[i] - y_pred
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

# Display final weights
print("Final Weights:", weights)
print("Final Bias:", bias)
