"""
    map, lambda, filter
"""

# 1 lambda 匿名函数 -> 可以作为回调函数
def add(x: float|int, y: float|int) -> float|int:
    return x+y

add_lambda = lambda x, y: x+y
print(f"普通函数：{add(1, 2)}")
print(f"匿名函数：{add_lambda(1, 2)}")

# 2 map -> 使用函数处理可迭代对象中的每一个元素，并返回一个新的可迭代对象
x: list = [1, 2, 3, 4, 5]
y: list = [6, 7, 8, 9, 10]
print(list(map(lambda a, b: a+b, x, y)))                  # 此处将map对象重新转换为list类型

# 3 filter -> 从可迭代对象中筛选出满足特定条件的元素（其中筛选的函数返回类型应该为布尔型）
origin = (x for x in range(1, 21))
print(list(filter(lambda x: x % 2, origin)))

