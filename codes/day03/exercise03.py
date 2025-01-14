"""
    面向对象
        多态
"""
import math

class Shape:
    def calculate_area(self):
        pass

class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def calculate_area(self) -> float:
        return self.base * self.height / 2

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def calculate_area(self) -> float:
        return (self.radius ** 2) * math.pi

class Rectangle(Shape):
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def calculate_area(self) -> float:
        return self.length * self.width

class Square(Rectangle):
    def __init__(self, length):
        Rectangle.__init__(self, length, length)

if __name__ == "__main__":
    triangle = Triangle(10, 5)
    circle = Circle(5)
    rectangle = Rectangle(10, 5)
    square = Square(5)

    print(f"三角形的面积为：{triangle.calculate_area()}\n"
          f"圆的面积为：{circle.calculate_area()}\n"
          f"长方形的面积为：{rectangle.calculate_area()}\n"
          f"正方形的面积为：{square.calculate_area()}\n")

