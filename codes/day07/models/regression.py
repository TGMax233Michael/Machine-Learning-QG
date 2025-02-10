import numpy as np
from preprocessing import polynomial_features

class MyLinearRegression:
    def __init__(self, 
                iterations=1000, 
                learning_rate=0.1, 
                print_info=False, 
                polynomial=False,
                degree=None,
                adagrad=False):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.polynomial = polynomial
        self.print_info = print_info
        self.degree = degree
        self.adagrad = adagrad
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
        weight = np.random.randn(n)
        
        if self.print_info:
            print(f"初始权重\n{weight}\n")
        
        for _ in range(self.iterations):
            # 计算均方误差
            y_pred = X.dot(weight)
            mse = np.mean((y_pred - y) ** 2)
                        
            # 梯度下降
            gradient = 2 / m * X.T.dot(y_pred - y)
            
            
            if self.adagrad == False:
                weight -= gradient * self.learning_rate
            else:
                r += gradient ** 2
                weight -= gradient * (self.learning_rate / np.sqrt(r) + 1e-8)
            
            if self.print_info:
                print(f"第{_+1}次迭代\n"
                      f"权重\n{weight}\n"
                      f"MSE {mse}\n\n")
            
        self.weight = weight
        self.mse = mse
        
        
        
    def predict(self, X):
        if self.polynomial:
            X = polynomial_features(X, self.degree)
        
        return X.dot(self.weight[1:]) + self.weight[0]