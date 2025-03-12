import numpy as np

def softmax(x):
    return np.e ** x / np.sum(np.e**x, axis=1)

data = np.array([1, 2, 3])

print(softmax(data))