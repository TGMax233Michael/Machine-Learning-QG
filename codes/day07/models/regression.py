import numpy as np
from preprocessing import polynomial_features, min_max_scaler
from enum import IntEnum, auto

class info_density(IntEnum):
    no = auto()
    few = auto()
    all = auto()

# TODO 增加梯度裁剪防止梯度爆炸
class MyLinearRegression:
    """线性模型

        属性:
            iterations (int): 迭代次数. 默认 1000
            learning_rate (float): 学习率. 默认 0.1
            print_info (info_density): 输出拟合过程信息. 默认 info_density.no
            polynomial (bool): 开启多项式拟合. 默认 False
            degree (int): 多项式拟合升阶数. 默认 None
            adagrad (bool): 开启Adagrad梯度算法. 默认 False
            epsilon (float): Adagrad算法中epsilon参数. 默认 1e-8
            mini_batch (bool): 是否开启小批量梯度下降法. 默认 False
        """
    def __init__(self, 
                iterations=1000, 
                learning_rate=0.1, 
                print_info=info_density.no,
                polynomial=False,
                degree=2,
                adagrad=False,
                epsilon=1e-8,
                mini_batch=False,
                adam = False,
                beta1 = 0.9,
                beta2 = 0.99
                ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.polynomial = polynomial
        self.print_info = print_info
        self.degree = degree
        self.adagrad = adagrad
        self.epsilon = epsilon
        self.mini_batch = mini_batch
        self.adam = adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight = None
        self.mse = None
        
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """拟合数据

        Args:
            X (np.ndarray): 特征
            y (np.ndarray): 标签
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        if self.polynomial:
            X = polynomial_features(X, self.degree)
            
        X = np.column_stack((np.ones(X.shape[0]), X))
        
        # 最大最小归一化
        # X = min_max_scaler(X, -10, 10)
            
        n_samples, n_features = X.shape[0], X.shape[1]
        
        if self.adagrad:
            r = np.ones(n_features)
            
        if self.adam:
            m = np.zeros(shape=(n_features,))
            v = np.zeros(shape=(n_features,))
        
        # 初始权重
        weight = np.zeros(n_features)
        
        if self.print_info != info_density.no:
            print(f"初始权重\n{weight}\n")
        
        for i in range(self.iterations):
            # 计算梯度
            if self.mini_batch:                 # 使用小批量梯度算法
                indices = np.random.choice(n_samples, n_samples//10, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]
                y_batch_pred = X_batch.dot(weight)
                gradient = 2 / (n_samples//10) * X_batch.T.dot(y_batch_pred - y_batch)
            else:                               # 使用批量梯度算法
                y_pred = X.dot(weight)
                gradient = 2 / n_samples * X.T.dot(y_pred - y)
            
            # 修正梯度爆炸
            clip_threshold = 5.0
            if np.linalg.norm(gradient) > clip_threshold:
                gradient = (clip_threshold / np.linalg.norm(gradient)) * gradient
            
            # 计算权重
            if self.adam:
                # Adam 优先
                m = self.beta1 * m + (1 - self.beta1) * gradient
                v = self.beta2 * v + (1 - self.beta2) * (gradient ** 2)
                m_hat = m / (1 - self.beta1 ** (i+1))
                v_hat = v / (1 - self.beta2 ** (i+1))
                weight -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            elif self.adagrad:
                # Adagrad
                r += gradient ** 2
                weight -= gradient * (self.learning_rate / (np.sqrt(r) + self.epsilon))
            else:
                # 普通梯度下降
                weight -= gradient * self.learning_rate

            y_pred_new = X.dot(weight)
            mse = np.mean((y_pred_new - y) ** 2)

            # 输出训练时的信息
            if self.print_info == info_density.few:
                if (i+1) % 10 == 0:
                    print(f"Iteration {i+1}/{self.iterations} MSE {mse}\n Weight {weight}\n\n")
            elif self.print_info == info_density.all:
                print(f"Iteration {i+1}/{self.iterations} MSE {mse}\n Weight {weight}\n\n")
            
        self.weight = weight
        self.mse = mse
        
        
        
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.polynomial:
            X = polynomial_features(X, self.degree)
        
        X = np.column_stack((np.ones(X.shape[0]), X))
        # X = min_max_scaler(X, 0, 1)
        
        return X.dot(self.weight)