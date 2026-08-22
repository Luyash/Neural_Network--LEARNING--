import numpy as np
import math


# Relu ---> negative and 0 return 0 natra return input number
def relu(x):
    return np.maximum(0, x)


# Sigmoid function --> y = 1/ ( 1+ e^(-x) )
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Binary Cross Entropy Loss Function (BCE):
# Used when outputs are 2 loke 0 and 1 or yes and no type
def BCE(actual, predicted):

    predicted = np.clip(predicted, 1e-15, 1 - 1e-15)

    loss = -(actual * np.log(predicted) +
             (1 - actual) * np.log(1 - predicted))

    return np.mean(loss) # Average returning

#==============
# DERIVATIVES
#==============

def derivative_sigmoid(x):
    derivative = sigmoid(x) * (1 - sigmoid(x) )
    return derivative


def derivative_BCE(actual, predicted):
    d_predicted = -(actual / predicted) + ((1 - actual) / (1 - predicted))
    return d_predicted

def derivative_reLu(x):
    return (x > 0).astype(float)