import numpy as np
from tqdm import trange

def sigmoid(X):
    return 1/(1+np.e ** (-X))
def softmax(X):
    return np.e**X / np.sum(np.e**X, axis=1, keepdims=True)

class BinaryLogisticRegression:
    """
        Binary Logistic Regression
        
        Attributes:
            n_epoches: 训练总轮数
            learning_rate: 学习率
    """
    def __init__(self, n_epoches=100, learning_rate=0.01):
        self.n_epoches = n_epoches
        self.learning_rate = learning_rate
        self.weights = None
        
    def _init_weights(self, n_features):
        self.weights =  np.zeros(shape=(n_features))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        
        for epoch in trange(self.n_epoches):
            y_pred = sigmoid(X.dot(self.weights))
            loss = -1/n_samples * (y*y_pred - np.log(1+np.e ** y_pred))
            gradient = -1/n_samples * X.T.dot(y-y_pred)
            self.weights -= gradient * self.learning_rate
            
    def predict(self, X: np.ndarray):
        y_pred = sigmoid(X.dot(self.weights))
        return np.where(y_pred > 0.5, 1, 0)
    
class LogisticRegression_ova:
    """
        Logistic Regression ova
        Attributes:
            n_epoches: 训练总轮数
            learning_rate: 学习率
    """
    def __init__(self, n_epoches=100, learning_rate=0.01):
        self.n_epoches = n_epoches
        self.learning_rate = learning_rate
        self.classifier: dict[int, BinaryLogisticRegression] = {}
        self.classes = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        
        for c in self.classes:
            y_binary = (y==c).astype(int)
            clf = BinaryLogisticRegression(n_epoches=self.n_epoches, learning_rate=self.learning_rate)
            clf.fit(X, y_binary)
            self.classifier[c] = clf
    
    
    def predict(self, X: np.ndarray):
        n_samples = X.shape[0]
        
        probabilities = np.zeros(shape=(n_samples, len(self.classes)))
        
        for i, c in enumerate(self.classes):
            probabilities[:, i] = self.classifier[c].predict(X)
            
        predictions = self.classes[np.argmax(probabilities, axis=1)]
        
        return predictions

class SoftmaxRegression:
    def __init__(self, n_epoches=100, learning_rate=0.01):
        self.n_epoches = n_epoches
        self.learning_rate = learning_rate
        self.classes = None
        self.weights = None
        
    def __init_weights(self, n_features):
        self.weights = np.zeros(shape=(n_features, len(self.classes)))
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        self.classes = np.arange(y.shape[1])
        # print(self.classes.shape)
        
        n_samples, n_features = X.shape
        
        self.__init_weights(n_features)
        # print(self.weights. shape)
        
        for epoch in trange(self.n_epoches):
            y_pred = X.dot(self.weights)
            # print(y_pred.shape)
            probs = softmax(y_pred)
            # print(y.shape)
            # print(probs.shape)
            
            loss = np.mean(-np.sum(y * np.log(probs), axis=1))
            gradient = -1/n_samples * X.T.dot(y - probs)
            self.weights -= gradient * self.learning_rate
            
            print(loss)
            
    def predict(self, X: np.ndarray):
        
        return np.argmax(softmax(X.dot(self.weights)), axis=1)

class LogisticRegression:
    """
        Attributes:
            n_epoches: 训练总轮数
            learning_rate: 学习率
            method: 包括两种方法 -> ["ova", "softmax]    
    """
    def __init__(self, n_epoches=100, learning_rate=0.01, method="ova"):
        self.n_epoches = n_epoches
        self.learning_rate = learning_rate
        self.method = method
        self.models_dict = {"ova": LogisticRegression_ova,
                            "softmax": SoftmaxRegression}
        self.model = None
        self._init_model()
        
    def _init_model(self):
        try:
            if self.method.lower() not in self.models_dict.keys():
                raise KeyError
        except KeyError:
            print(f"Unknown Methods! Existed Methods: {self.models_dict.keys()}")
        self.model = self.models_dict[self.method.lower()](self.n_epoches, self.learning_rate)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
            