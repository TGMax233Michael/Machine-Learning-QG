"""
        泰坦尼克号数据集分析

        发现前面学的还不够多 例如下面用到的pandas的groupby 还有matplotlib的pie和bar 捂脸/
        算是边学边做的吧
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("train.csv", index_col = "PassengerId")

# 基本信息
# print(data.head())
# print(data.sample(5))
# print(data.describe())
# print(data.info())
# print()

label_death = ["Survived", "Death"]

# 死亡统计
print(data["Survived"].value_counts())


# 可视化
plt.figure(figsize=(8, 8), dpi = 40)
# plt.pie() 绘制饼状图
# autopct -> 显示方式（类似c语言中的字符串格式）
# pctdistance -> 百分比与饼图中心的距离
# labeldistance -> 标签与饼图中心的距离
# explode -> 向外突出显示(列表中的值对应x中的不同标签)
plt.pie(data["Survived"].value_counts(), autopct='%.2f%%', labels=label_death,
        pctdistance=0.4, labeldistance=0.7, explode=[0,0.1], shadow=True)
plt.title("Survived Ratio")


plt.waitforbuttonpress()
plt.close()



# 按照性别的死亡统计
by_sex = data.groupby(by="Sex")["Survived"]
print(f"{by_sex.get_group("male").reset_index()}\n")
print(f"{by_sex.get_group("female").reset_index()}\n")
print(f"{by_sex.value_counts()}\n")


# 可视化
plt.figure(figsize=(16, 8), dpi = 80)
plt.subplot(1, 2, 1)
plt.pie(by_sex.value_counts()["male"], autopct="%.2f%%", labels=label_death,
        pctdistance=0.4, labeldistance=0.7, explode=[0,0.1], shadow=True)
plt.title("Male Survived Ratio")

plt.subplot(1, 2, 2)
plt.pie(by_sex.value_counts()["female"], autopct="%.2f%%", labels=label_death,
        pctdistance=0.4, labeldistance=0.7, explode=[0,0.1], shadow=True)
plt.title("Female Survived Ratio")


plt.waitforbuttonpress()
plt.close()



# 年龄的死亡统计
by_age = data["Age"]
print(by_age)
print(f"min_age = {by_age.min()} | max_age = {by_age.max()}") # [0, 80]

# numpy.histogram() 计算数据集的直方图
# range -> 范围
# bins -> 分割的份数/柱子个数
age_num,_ = np.histogram(by_age, range=(0, 80), bins=16)

age_survived = []
for age in range(5, 81, 5):
    # 筛选数据 (loc除了能根据行列标签来定位数据，也可以使用布尔表达式)
    filter_data = data.loc[(by_age >= age - 5) & (by_age <= age)]['Survived']
    survived_num = filter_data.sum()
    age_survived.append(survived_num)


# 可视化
plt.figure(figsize=(10, 5), dpi=60)

# plt.bar() 绘制柱状图
# x -> x轴(浮点数序列)
# height -> 高度(浮点数序列)
# width -> 柱宽
plt.bar(np.arange(2,78,5)+0.5, age_num, width=5, label='Total')
plt.bar(np.arange(2,78,5)+0.5, age_survived, width=5, label='Survived')
plt.xlabel("Age")
plt.legend()
plt.ylabel("Survival")
plt.title("Survival - Age")


plt.waitforbuttonpress()
plt.close()

# 登录港口的死亡统计
by_embark = data.groupby(by=["Embarked"])["Survived"]
print(f"{by_embark.get_group("S")}\n")
print(f"{by_embark.get_group("C")}\n")
print(f"{by_embark.get_group("Q")}\n")
print(f"{by_embark.value_counts()}\n")

# 可视化
plt.figure(figsize=(15, 5), dpi=80)
plt.subplot(1, 3, 1)
plt.pie(by_embark.value_counts()["S"], autopct="%.2f%%",
        pctdistance=0.2, labeldistance=0.6, shadow=True,
        explode=[0.05, 0.05], labels=label_death, colors=["GREEN", "RED"])
plt.title("Embark on S")


plt.subplot(1, 3, 2)
plt.pie(by_embark.value_counts()["C"], autopct="%.2f%%",
        pctdistance=0.2, labeldistance=0.6, shadow=True,
        explode=[0.05, 0.05], labels=label_death, colors=["GREEN", "RED"])
plt.title("Embark on C")


plt.subplot(1, 3, 3)
plt.pie(by_embark.value_counts()["Q"], autopct="%.2f%%",
        pctdistance=0.2, labeldistance=0.4, shadow=True,
        explode=[0.05, 0.05], labels=label_death, colors=["GREEN", "RED"])
plt.title("Embark on Q")


plt.waitforbuttonpress()
plt.close()

# 家庭大小的死亡统计
by_family_size = data["SibSp"] + data["Parch"] + 1
print(f"{by_family_size.value_counts()}")   # [1, 11]

family_size_num, _ = np.histogram(by_family_size, range=(1, 11), bins=11)

family_size_survived = []

for size in range(1, 12):
    filter_data_size = data.loc[(by_family_size == size), "Survived"]
    survived_num = filter_data_size.sum()
    family_size_survived.append(survived_num)

plt.figure(figsize=(8, 8), dpi=60)
plt.bar(np.arange(1, 12), family_size_num, width=0.8, label="Total")
plt.bar(np.arange(1, 12), family_size_survived, width=0.8, label="Survived")
plt.legend()
plt.xlabel("Total")
plt.ylabel("Survived")
plt.xticks(np.arange(1, 12))
plt.title("Survived - Family Size")
plt.grid(True, linewidth=0.4)


plt.waitforbuttonpress()
plt.close()


