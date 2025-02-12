import numpy as np
import pandas as pd
from preprocessing import min_max_scaler
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("BostonHousing.csv")
data = data.dropna(axis=0, how="any")
print(data.iloc[10, 5])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X = np.array(X)
y = np.array(y)
print(f"X[10, 5]: {X[10, 5]}")
X = min_max_scaler(X)
print(X)

# arr = np.arange(1, 11).reshape(5, 2)
# print(arr)
# poly = PolynomialFeatures(3)
# print(poly.fit_transform(arr))