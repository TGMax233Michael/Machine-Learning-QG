# 基本数据类型
a: int = 1
b: float = 3
c: bool = True
c = (a > b)                         # False
d: complex = complex(1, 3)          # 1+3i
d_conj: complex = d.conjugate()     # 1-3i（求共轭复数）
list_res: list = [a, b, c, d, d_conj]
print("1. 基本数据类型")
for x in list_res:
    print(f'类型为{str(type(x)):<20} | 结果为{x:<8}')
print()

# 容器数据类型
text: str = "I Love Python"
list_test: list = []
hash_table: dict = {"a": 1, "b": 2} # Python中哈希表实现 -> 键值对
tuple_test: tuple = (0, 1, 2)
set_test: set = set()               # 集合中不会有重复元素
set_test.add(4)
set_test.add(1)
set_test.add(2)
set_test.add(3)
set_test.add(1)
list_res = [text, list_test, hash_table, tuple_test, set_test]
print("2. 容器")
for x in list_res:
    print(f'类型为{str(type(x)):<20} | 结果为{str(x):<20}')

