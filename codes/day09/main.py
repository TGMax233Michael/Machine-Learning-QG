import numpy as np
from sklearn.datasets import make_classification
from model.classification import LogisticRegression

def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    acc_matrix = (y_true == y_pred)
    return np.sum(acc_matrix) / acc_matrix.shape[-1]

if __name__ == "__main__":
    data = make_classification(n_samples=100, n_classes=2, random_state=42, n_features=2, n_redundant=0, n_repeated=0)

    x, y = data[0], data[1]

    model = LogisticRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    print(f"true: {y}")
    print(f"predict: {y_pred}")
    
    print("Accuracy: {accuracy(y, y_pred)}")
