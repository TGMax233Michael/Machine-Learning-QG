"""
    pandas 读取csv与存储
"""
import pandas as pd
data01 = pd.read_csv("../day06/financial_data.csv",
                   index_col=0,     # index_col将某一列设置为行索引
                   header = 0,      # header 跳过几行再读取数据
                   )

print(data01)

data02 = pd.DataFrame({"姓名": ["沃尔斯", "诺柒", "阿卡姆"],
                       "种族": ["狼", "虎", "犬"],
                       "喜好": ["澳白", "草莓蛋糕", "抹茶冰淇淋"]})
print(data02)

data02.to_csv("characters.csv")

