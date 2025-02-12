from models.regression import MyLinearRegression, info_density
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

def show_result(model, name=None):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    if name:
        print(name)
    else:
        print(model.__class__.__name__)
    
    print(f"Train MSE: {mean_squared_error(y_train, y_train_pred)}")
    print(f"Train R2: {r2_score(y_train, y_train_pred)}")    
    print(f"Test MSE: {mean_squared_error(y_test, y_test_pred)}")
    print(f"Test R2: {r2_score(y_test, y_test_pred)}")
    print()


# 1. 数据预处理
data = pd.read_csv("BostonHousing.csv")

# rm 存在 5 个缺失值，数量较少采取直接删除样本
data = data.dropna(axis=0, how="any")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# print(X.isna().sum())
# print(y.isna().sum())
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
                            learning_rate=0.01, 
                            polynomial=True, 
                            degree=2, 
                            iterations=10000, 
                            print_info=info_density.no,
                            adagrad=False,
                            mini_batch=True,
                            adam=True
                        )

models = {
    "SkLearn LinearRegression" : sklinear_model,
    "My Linear Regressin" : my_linear_model,
    "Sklearn Pipline(PolynomialFeatures(degree=2), LinearRegression)" : skpoly_model,
    "My Polynomial Regression (degree=2)" : my_poly_model
}


# 3. 模型评估
for name, model in models.items():
    show_result(model, name)
