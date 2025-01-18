---
Field:
    - 机器学习/预测
,    - 数据挖掘

Ext:
    - .csv

DatasetUsage:
    - 89823
---

# **背景介绍** 

Titanic数据集是非常适合数据科学和机器学习新手入门练习的数据集。
数据集为1912年泰坦尼克号沉船事件中一些船员的个人信息以及存活状况。这些历史数据已经非分为训练集和测试集，你可以根据训练集训练出合适的模型并预测测试集中的存活状况。

### 如何在线使用数据集

创建项目后：
Python用户，输入`ls ../input/titanic/`   查看数据路径
R用户，输入`list.files("../input/titanic/")`  查看数据路径
使用相关包读取数据

# **数据描述** 

# 数据字典

| **变量名**   | PassengerId | Survived                 | Pclass                                 | Name     | Sex         | Age         | SibSp                    | Parch                | Ticket   | Fare    | Cabin  | Embarked                                                      |
|----------|-------------|--------------------------|----------------------------------------|----------|-------------|-------------|--------------------------|----------------------|----------|---------|--------|---------------------------------------------------------------|
| **变量解释** | 乘客编号    | 乘客是否存活(0=NO 1=Yes) | 乘客所在的船舱等级,(1=1st,2=2nd,3=3rd) | 乘客姓名 | 乘客性别    | 乘客年龄    | 乘客的兄弟姐妹和配偶数量 | 乘客的父母与子女数量 | 票的编号 | 票价    | 座位号 | 乘客登船码头。 C = Cherbourg; Q = Queenstown; S = Southampton |
| **数据类型** | numeric     | categorical              | categorical                            | string   | categorical | categorical | numeric                  | numeric              | string   | numeric | string | categorical                                                   |

***

# 数据预览

`train.csv ` 大小：60k

| PassengerId | Survived | Pclass | Name                                                | Sex    | Age | SibSp | Parch | Ticket           | Fare    | Cabin | Embarked |
|-------------|----------|--------|-----------------------------------------------------|--------|-----|-------|-------|------------------|---------|-------|----------|
| 1           | 0        | 3      | Braund, Mr. Owen Harris                             | male   | 22  | 1     | 0     | A/5 21171        | 7.25    |       | S        |
| 2           | 1        | 1      | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female | 38  | 1     | 0     | PC 17599         | 71.2833 | C85   | C        |
| 3           | 1        | 3      | Heikkinen, Miss. Laina                              | female | 26  | 0     | 0     | STON/O2. 3101282 | 7.925   |       | S        |
| 4           | 1        | 1      | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | female | 35  | 1     | 0     | 113803           | 53.1    | C123  | S        |
| 5           | 0        | 3      | Allen, Mr. William Henry                            | male   | 35  | 0     | 0     | 373450           | 8.05    |       | S        |
| 6           | 0        | 3      | Moran, Mr. James                                    | male   |     | 0     | 0     | 330877           | 8.4583  |       | Q        |
| 7           | 0        | 1      | McCarthy, Mr. Timothy J                             | male   | 54  | 0     | 0     | 17463            | 51.8625 | E46   | S        |
| 8           | 0        | 3      | Palsson, Master. Gosta Leonard                      | male   | 2   | 3     | 1     | 349909           | 21.075  |       | S        |
| 9           | 1        | 3      | Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)   | female | 27  | 0     | 2     | 347742           | 11.1333 |       | S        |
| 10          | 1        | 2      | Nasser, Mrs. Nicholas (Adele Achem)                 | female | 14  | 1     | 0     | 237736           | 30.0708 |       | C        |
| …           | …        | …      | …                                                   | …      | …   | …     | …     | …                | …       | …     | …        |

**10** of  of` 891` rows, `12` columns 



> 如果您对数据集有疑问或建议，可以提交意见反馈给我们哦 。

## **引用格式**
```
@misc{titanic,
    title = { Titanic数据集 },
    author = { 小鲸 },
    howpublished = { \url{https://www.heywhale.com/mw/dataset/58a940107159a710d916aefb} },
    year = { 2017 },
}
```