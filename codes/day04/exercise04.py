"""
    pandas 基本数据处理
"""
import pandas as pd

data = pd.read_csv("financial_data.csv", index_col=0)

print(data)
print()
# 1. 排序
data.sort_values(by="总支出", inplace=True, ascending=False)
print(data)
print()

data.sort_values(by=["食品", "饮料"], inplace=True, ascending=[True, False])
print(data)