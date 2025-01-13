# 1 基本数据类型
# 1.1 整型
a: int = 1

# 1.2 浮点型
b: float = 3

# 1.3 布尔型
c: bool = True
c = (a > b)                         # False

# 1.4 复数型
d: complex = complex(1, 3)          # 1+3i
d_conj: complex = d.conjugate()     # 1-3i（求共轭复数）
list_res: list = [a, b, c, d, d_conj]

print("1. 基本数据类型")
for x in list_res:
    print(f'类型为{str(type(x)):<20} | 结果为{x:<8}')
print()

# 2 容器数据类型
# 2.1 字符串
text: str = "I Love Python"

# 2.2 列表
list_test: list = [10]
list_test.insert(0, 12)
for __ in range(0, 10):
    list_test.append(__)
list_test.sort(reverse=True)
del list_test[0]
list_test2 = list_test[-1::-1]

# 2.3 字典
hash_table: dict = {"a": 1, "b": 2} # Python中哈希表实现

# 2.4 元组
tuple_test: tuple = (0, 1, 2)

# 2.5 集合
set_test: set = set()               # 集合中不会有重复元素
set_test.add(4)
set_test.add(1)
set_test.add(2)
set_test.add(3)
set_test.add(1)


list_res = [text, list_test, list_test2, hash_table, tuple_test, set_test]
print("2. 容器")
for x in list_res:
    print(f'类型为{str(type(x)):<20} | 结果为{str(x):<20}')

