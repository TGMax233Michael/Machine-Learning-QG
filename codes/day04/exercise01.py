"""
    numpy库
        基于列表构建矩阵 特殊矩阵构建 矩阵乘法
        矩阵广播机制 矩阵转置 矩阵的逆 矩阵存取
"""
import numpy as np

# 1. 列表构建矩阵
matrix01 = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=np.int8)

print(matrix01)
print(type(matrix01))

# 2. 特殊矩阵构建方法
# 2.1 元素相同
matrix_zeros = np.zeros(shape=(2, 3), dtype=np.int8)
matrix_ones = np.ones(shape=(4, 5), dtype=np.int8)
matrix_pi = np.full((10, 10), np.pi)

# 2.2 范围
arr1 = np.arange(1, 10, 2)
arr2 = np.linspace(1, 10, 100)
arr2 = arr2.reshape(10, 10)     # 转为10*10的矩阵

print(matrix_zeros)
print(matrix_ones)
print(matrix_pi)
print(arr1)
print(arr2)

# 3. 矩阵乘法
matrix02 = np.array([2, 3, 1]).reshape(-1, 1)
translate_matrix = np.array([[1, 0, 20],
                             [0, 1, 20],
                             [0, 0, 1]])
print(translate_matrix.dot(matrix02))

matrix03 = np.arange(1, 10).reshape(-1, 3)
print(matrix03.dot(matrix03))

# 4. 矩阵的广播机制
matrix04 = np.ones(shape=(3, 3))
matrix04 += 4
print(matrix04)
matrix04 *= 4
print(matrix04)
matrix04 -= 9
print(matrix04)

# 5. 矩阵的转置
matrix05 = np.arange(1, 10).reshape(-1, 3)
print(matrix05)
print(matrix05.T)

# 6. 矩阵的存取
matrix06 = np.arange(1, 26).reshape(-1, 5)
print(matrix06[1, 2])
matrix06[3, :] = [1, 2, 3, 4, 5]
print(matrix06)