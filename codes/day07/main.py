from models.regression import MyLinearRegression
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd

# california_housing = fetch_california_housing(as_frame=True)
# data = california_housing.frame

# print(data.describe().T)
# print(data.sample(5).T)

X, y = make_regression(n_samples=1000, n_features=6, noise=1, random_state=42)

# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]

# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("sklearn")
print(f"mse: {mean_squared_error(y_test, y_pred)}")
print(f"r2: {r2_score(y_test, y_pred)}")

mymodel = MyLinearRegression(iterations=100, print_info=False)
mymodel.fit(X_train, y_train)
y_pred = mymodel.predict(X_test)
print("mine")
print(f"mse: {mean_squared_error(y_test, y_pred)}")
print(f"r2: {r2_score(y_test, y_pred)}")