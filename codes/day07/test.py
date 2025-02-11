import numpy as np
import pandas as pd
from preprocessing import min_max_scaler

data = pd.read_csv("BostonHousing.csv")
data = data.dropna()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X = np.array(X)
y = np.array(y)
X = min_max_scaler(X)
print()