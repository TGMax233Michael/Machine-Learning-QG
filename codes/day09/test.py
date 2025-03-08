import numpy as np
from sklearn.datasets import make_classification

sigmoid = lambda x: (1 / (1 + np.e ** (-x)))

def judge(x):
    return np.where(x > 0.5, 1, 0)

data = make_classification(n_samples=10, n_classes=2, random_state=42, n_features=2, n_redundant=0, n_repeated=0)
x, y = data[0], data[-1]
# print(x)
# print(y)

learning_rate = 0.01
# Logsitic Regression
features = x.shape[1]
samples = x.shape[0]
# print(features, samples)

weights = np.zeros(shape=(features))
# print(weights)

for epoch in range(1000):
    y_pred = sigmoid(x.dot(weights))
    loss = - 1/samples * ((y*y_pred) - np.log(1+np.e ** (y_pred)))
    # print(y_pred)
    gradient = -1/samples * x.T.dot(y-y_pred)
    # print(gradient)

    weights -= gradient * learning_rate
    print(loss)

print()
print(judge(y_pred))
print(y)