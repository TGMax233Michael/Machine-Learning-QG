import numpy as np
from tqdm import trange

def sigmoid(x):
    return 1/(1+np.e ** (-x))

class LogisticRegression:
    def __init__(self, n_epoches=100, learning_rate=0.01):
        self.n_epoches = n_epoches
        self.learning_rate = learning_rate
        self.weights = None
        self.featues = None
        self.samples = None
        
    def _init_weights(self):
        self.weights =  np.zeros(shape=(self.featues))
    
    def _calc_gradient(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        return -1/self.samples * (x.dot(y-y_pred))
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        self.samples, self.featues = x.shape[0], x.shape[1]
        self._init_weights()
        
        for i in trange(self.n_epoches):
            y_pred = sigmoid(x.dot(self.weights))
            loss = -1/self.samples * (y*y_pred - np.log(1+np.e ** y_pred))
            gradient = -1/self.samples * x.T.dot(y-y_pred)
            self.weights -= gradient * self.learning_rate
            
    def predict(self, x: np.ndarray):
        y_pred = sigmoid(x.dot(self.weights))
        return np.where(y_pred > 0.5, 1, 0)
            
    