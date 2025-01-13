"""
    推导式
"""
# 1 列表推导式
list_square: list = [x**2 for x in range(0, 10)]
print(list_square)
print()

# 2 集合推导式
set_square: set = {x**2 for x in range(0, 10)}
print(set_square)
print()

# 3 字典推导式
dict_square: dict = {x: x**2 for x in range(0, 10)}
print(dict_square)
print()

# 4 嵌套推导式
matrix_square: list = [[x**2 for x in range(i*10, (i+1)*10)] for i in range(0, 10)]
for i in range(0, 9):
    print(matrix_square[i])
print()