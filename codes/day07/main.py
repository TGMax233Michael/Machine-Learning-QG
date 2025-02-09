from models.regression import MyLinearRegression, MyPolynomialRegression
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from preprocessing import polynomial_features
import pandas as pd

# california_housing = fetch_california_housing(as_frame=True)
# data = california_housing.frame

# print(data.describe().T)
# print(data.sample(5).T)

X, y = make_regression(n_samples=100, n_features=3, noise=1, random_state=42)

# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]

# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sklinear_model = LinearRegression()
sklinear_model.fit(X_train, y_train)
y_pred = sklinear_model.predict(X_test)
print("sklearn linear")
print(f"mse: {mean_squared_error(y_test, y_pred)}")
print(f"r2: {r2_score(y_test, y_pred)}")
print()

my_linear_model = MyLinearRegression(iterations=100, print_info=False)
my_linear_model.fit(X_train, y_train)
y_pred = my_linear_model.predict(X_test)
print("my linear")
print(f"mse: {mean_squared_error(y_test, y_pred)}")
print(f"r2: {r2_score(y_test, y_pred)}")
print()

my_polynomial_model = MyPolynomialRegression(degree=2, iterations=100, print_info=False)
my_polynomial_model.fit(X, y)
y_pred = my_linear_model.predict(X_test)
print("my polynomial")
print(f"mse: {mean_squared_error(y_test, y_pred)}")
print(f"r2: {r2_score(y_test, y_pred)}")
print()

print(polynomial_features(X, degree=2))