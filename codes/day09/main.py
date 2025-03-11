import numpy as np
from sklearn.datasets import make_classification
from model.classification import BinaryLogisticRegression, LogisticRegression

def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    acc_matrix = (y_true == y_pred)
    return np.mean(np.sum(acc_matrix))

if __name__ == "__main__":
    data = make_classification(n_samples=300, n_classes=4, random_state=42, n_informative=6, n_redundant=0, n_repeated=0)

    x, y = data[0], data[1]

    model = LogisticRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    print(f"true: {y}")
    print(f"predict: {y_pred}")
    
    print(f"Accuracy: {accuracy(y, y_pred)}")
