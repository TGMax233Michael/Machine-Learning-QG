"""
    面向对象
        继承
"""

# 1 继承
# 1.1 单继承
# 父类
class Animal:
    def __init__(self, name):
        self.name = name

    def eat(self):
        print(f"{self.name} is eating.")

# 子类
class Dog(Animal):
    def bark(self):
        print(f"{self.name} is barking.")

puppy = Dog("旺财")
puppy.eat()
puppy.bark()

# 1.2 多继承
class Father:
    def __init__(self, name):
        self.father_name = name

class Mother:
    def __init__(self, name):
        self.mother_name = name

class Child(Father, Mother):
    def __init__(self, child_name, father_name, mother_name):
        Father.__init__(self, father_name)
        Mother.__init__(self, mother_name)
        self.name = child_name

    def show_info(self):
        print(f"I'm {self.name}\n"
              f"My father is {self.father_name}\n"
              f"My mother is {self.mother_name}")

print()
John = Child("A", "B", "C")
John.show_info()



