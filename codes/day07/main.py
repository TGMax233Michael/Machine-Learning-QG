from models.regression import MyLinearRegression, info_density
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from preprocessing import polynomial_features
import pandas as pd
import numpy as np

def show_result(model, name=None):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if name:
        print(name)
    else:
        print(model.__class__.__name__)
        
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"R2: {r2_score(y_test, y_pred)}")
    print()


# 1. 数据预处理
data = pd.read_csv("BostonHousing.csv")
# print(X.isna().sum())
# print(y.isna().sum())

# rm 存在 5 个缺失值，数量较少采取直接删除样本
data = data.dropna()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X = np.array(X)
y = np.array(y)

# X, y = make_regression(n_samples=1000, n_features=8, n_informative=6, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. 学习过程
sklinear_model = LinearRegression()

my_linear_model = MyLinearRegression(
                            learning_rate=0.01, 
                            iterations=10000, 
                            print_info=info_density.no,
                            adagrad=False,
                            mini_batch=True,
                            adam=True
                        )

skpoly_model = make_pipeline(PolynomialFeatures(degree=2), 
                             LinearRegression())

my_poly_model = MyLinearRegression(
                            learning_rate=0.1, 
                            polynomial=True, 
                            degree=2, 
                            iterations=10000, 
                            print_info=info_density.no,
                            adagrad=False,
                            mini_batch=False,
                            adam=True
                        )

models = [sklinear_model, my_linear_model, skpoly_model, my_poly_model]


# 3. 模型评估
for model in models:
    show_result(model)
