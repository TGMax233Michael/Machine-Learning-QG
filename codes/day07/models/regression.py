import numpy as np
from preprocessing import polynomial_features
from enum import IntEnum, auto

class info_density(IntEnum):
    no = auto()
    few = auto()
    all = auto()

# TODO 增加梯度裁剪防止梯度爆炸
class MyLinearRegression:
    def __init__(self, 
                iterations=1000, 
                learning_rate=0.1, 
                print_info=info_density.no,
                polynomial=False,
                degree=None,
                adagrad=False,
                epsilon=1e-8,
                mini_batch=False,
                ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.polynomial = polynomial
        self.print_info = print_info
        self.degree = degree
        self.adagrad = adagrad
        self.epsilon = epsilon
        self.mini_batch = mini_batch
        self.weight = None
        self.mse = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        if self.polynomial:
            X = polynomial_features(X, self.degree)
            
        X = np.column_stack((np.ones(X.shape[0]), X))
            
        m, n = X.shape[0], X.shape[1]
        
        if self.adagrad:
            r = np.ones(n)
        
        # 初始权重
        weight = np.zeros(n)
        
        if self.print_info != info_density.no:
            print(f"初始权重\n{weight}\n")
        
        for _ in range(self.iterations):
            if self.mini_batch:
                indices = np.random.choice(m, m//10, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
                y_batch_pred = X_batch.dot(weight)
                gradient = 2 / (m//10) * X_batch.T.dot(y_batch_pred - y_batch)
            else:
                y_pred = X.dot(weight)
                gradient = 2 / m * X.T.dot(y_pred - y)
            
            if not self.adagrad:
                weight -= gradient * self.learning_rate
            else:
                r += gradient ** 2
                weight -= gradient * (self.learning_rate / (np.sqrt(r) + self.epsilon))
            
            y_pred_new = X.dot(weight)
            mse = np.mean((y_pred_new - y) ** 2)

            if self.print_info == info_density.few:
                if _ % 5 == 0:
                    print(f"第{_+1}次迭代\n"
                        f"权重\n{weight}\n"
                        f"MSE {mse}\n\n")
            elif self.print_info == info_density.all:
                print(f"第{_+1}次迭代\n"
                        f"权重\n{weight}\n"
                        f"MSE {mse}\n\n")
            
        self.weight = weight
        self.mse = mse
        
        
        
    def predict(self, X):
        if X.ndim == 1:
            X.reshape(-1, 1)
        
        if self.polynomial:
            X = polynomial_features(X, self.degree)
        
        return X.dot(self.weight[1:]) + self.weight[0]  