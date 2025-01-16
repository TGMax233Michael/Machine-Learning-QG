"""
    pandas
"""
import pandas as pd
import numpy as np

sr1 = pd.Series([1, 2, 3, 4], index = [1, 2, 3, 4], name="data")
sr1.name = "xd"
print(sr1)
print(sr1.index)
print(sr1.loc[2])
print(sr1.iloc[0])

# 利用字典初始化Series
sr2 = pd.Series({"d1": 1,
                 "d2": 2,
                 "d3": 3}, name="sr2")
print(sr2)

# DataFrame
df = pd.DataFrame(np.arange(0, 9).reshape(-1, 3), index=[1, 2, 3], columns=["s1", "s2", "s3"])
print(df)