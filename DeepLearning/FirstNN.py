import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x * (1-x)
X = np.array([[0,1,0],
              [1,1,1],
              [1,0,0],
              [0,1,1]])
Y = np.array([[1,0,1,0]]).T
np.random.seed(1)
bias = 2 * np.random.random((3,1)) -1
print('Bias equal to: ')
print(bias)
for i in range(1000):
    inputs_layer = X
    outputs = sigmoid(np.dot(inputs_layer,bias))
    error = Y - outputs
    adjustments = error * sigmoid_derivative(outputs)
    bias += np.dot(X.T,adjustments)
print('Bias after training: ')
print(bias)
print('Error is: ')
print(error)
print('Output after training: ')
print(outputs)
