import zipfile
import numpy as np
from functions import *

with zipfile.ZipFile(
    "Breast Cancer/breast+cancer+wisconsin+diagnostic.zip",
    "r"
) as zip_file:
    zip_file.extract("wdbc.data")

data = np.genfromtxt(
    "wdbc.data",
    delimiter=",",
    dtype=str,
)

# Getting data Inputs and Output:
X = data[:, 2:32].astype(float)  # columns 2 through 31 → 30 inputs
y = np.where(data[:, 1:2] == "M", 1, 0).astype(float)    # column 1 → diagnosis

# Adding randomizer for the input data like biccha biccha bata 80% and 20%
indices = np.random.permutation(len(X))

X = X[indices]
y = y[indices]

# Splitting the data into 80% training and 20% test
split_index = int(len(X) * 0.8)  # Gives 80% vaneko kati ho vanera 

X_train = X[:split_index] # First 80%
X_test = X[split_index:]  # Remaining 20%

y_train = y[:split_index]  # First 80%
y_test = y[split_index:]   # Remaining 20%

# Calculate mean, SD for Standardization:
train_mean = X_train.mean(axis=0)
train_sd = X_train.std(axis=0)
    
# Standardize
X_train_scaled = (X_train - train_mean) / train_sd
X_test_scaled = (X_test - train_mean) / train_sd


# Now the actual fun part:

# Initializing weights and biases:
W1 = np.random.randn(30, 16) * np.sqrt(2 / 30)  # Random values but around +-1
b1 = np.zeros(16)

W2 = np.random.randn(16, 8) * np.sqrt(2 / 16) # This kind of initialization is called He Initialization This is good for Relu activation function
b2 = np.zeros(8)

W3 = np.random.randn(8, 1) # No He initialization here because aaba Sigmoid lai dine not ReLu; rakhda nee kei huna chhai hudaina huna chhai 
b3 = np.zeros(1)

learning_rate = 0.001
batch_size = 32

# The main loop

for epoch in range (1,20000,1):

    # Forward Propagation:

    # Layer 1
    intermediate_output1 = X_train_scaled @ W1 + b1
    output_1 = relu(intermediate_output1)

    # Layer 2
    intermediate_output2 = output_1 @ W2 + b2
    output_2 = relu(intermediate_output2)

    #Layer 3 
    intermediate_output3 = output_2 @ W3 + b3
    predicted = sigmoid(intermediate_output3)

    # Loss for the predicted values:
    loss = BCE(y_train , predicted)

    # ======================
    # BACKPROPAGATION
    # =====================

    # Output layer
    # BCE + Sigmoid simplifies to:
    d_intermediate_output3 = predicted - y_train

    # Gradients for W3 and b3
    dW3 = output_2.T @ d_intermediate_output3
    db3 = np.sum(d_intermediate_output3, axis=0)


    # Send gradient backwards through W3
    d_output2 = d_intermediate_output3 @ W3.T

    # ReLU derivative
    d_intermediate_output2 = d_output2 * derivative_reLu(intermediate_output2)

    # Gradients for W2 and b2
    dW2 = output_1.T @ d_intermediate_output2
    db2 = np.sum(d_intermediate_output2, axis=0)


    # Send gradient backwards through W2
    d_output1 = d_intermediate_output2 @ W2.T

    # ReLU derivative
    d_intermediate_output1 = d_output1 * derivative_reLu(intermediate_output1)

    # Gradients for W1 and b1
    dW1 = X_train_scaled.T @ d_intermediate_output1
    db1 = np.sum(d_intermediate_output1, axis=0)


    # Gradient Descent
    # ==============================
    # UPDATING WEIGHTS AND BIASES
    # ==============================

    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    W1 = W1 - learning_rate * dW1
    b1 - b1 - learning_rate * db1



# =================
# TESTING
# =================

# Layer 1
intermediate_output1 = X_test_scaled @ W1 + b1
output_1 = relu(intermediate_output1)

# Layer 2
intermediate_output2 = output_1 @ W2 + b2
output_2 = relu(intermediate_output2)

# Layer 3
intermediate_output3 = output_2 @ W3 + b3
predicted = sigmoid(intermediate_output3)

# Test loss
loss = BCE(y_test, predicted)

print(f"The test loss for this network is --> {loss}")
    











