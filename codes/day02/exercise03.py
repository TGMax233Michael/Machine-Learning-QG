"""
    异常处理
    常见的异常包括：SyntaxError, IndexError, NameError, TypeError, IndentationError, etc
"""

list_item = ["武装直升机", "沃尔玛购物袋", "钝角", "QG工作室牛逼"]

# 样例1 -> 索引异常
index: int = int(input("请输入索引："))
try:
    if index-1 < 0:
        raise IndexError
    print(list_item[index-1])
except IndexError as error:
    print(f"发生错误：{error}")
    print("索引超出了范围([1, 4])")


# 样例2 -> 0不能作除数
div1: float = float(input("请输入被除数："))
div2: float = float(input("请输入除数："))
try:
    print(div1 / div2)
except ZeroDivisionError as error:
    print(f"发生错误: {error}")
    print("除数不能为0！")