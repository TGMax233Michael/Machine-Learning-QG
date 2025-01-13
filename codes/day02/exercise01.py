"""
条件语句与循环语句
"""

# 1 条件语句
gender: str = input("请输入性别：")

if gender == "男":
    print("男生")
elif gender == "女":
    print("女生")
else:
    print("武装直升机/沃尔玛购物袋")
print()

# 2 循环语句
# 2.1 while循环
a = 1
while a < 10:
    if a % 2 == 0:
        print(a)
    a += 1
print()

# 2.2 for循环（可以用来遍历可迭代对象）
list_item = ["武装直升机", "沃尔玛购物袋", "钝角", "QG工作室牛逼"]
for item in list_item:
    print(item)
print()

for x in range(2, 21, 2):
    print(x)
print()
