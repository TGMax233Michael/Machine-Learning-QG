from models.regression import MyLinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
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

data = pd.read_csv("BostonHousing.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# print(X.isna().sum())
# print(y.isna().sum())

# rm 存在 5 个缺失值，数量较少采取直接删除样本
X = X.dropna()

X = np.array(X)
y = np.array(y)


X, y = make_regression(n_samples=1000, n_features=3, noise=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sklinear_model = LinearRegression()
my_linear_model = MyLinearRegression(iterations=1000, print_info=False)
skpoly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
my_poly_model = MyLinearRegression(polynomial=True, degree=2, iterations=1000, print_info=False)
models = [sklinear_model, my_linear_model, skpoly_model, my_poly_model]

for model in models:
    show_result(model)
