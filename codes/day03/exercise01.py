"""
    面向对象
        类与实例化 类变量以及方法 私有变量和私有方法
"""

class ListNode:
    def __init__(self, value = 0, next = None):
        self.value = value
        self.next: ListNode | None = next

class LinkedList:
    what_the_fuck = 1   # -> 类变量
    def __init__(self):
        self.head: ListNode | None = None # -> 私有变量

    # 类方法
    def initialize_linked_list(self):
        last = None

        for i in range(0, 10):
            newNode = ListNode(i)

            if self.head:
                last.next = newNode
            else:
                self.head = newNode

            last = newNode

    def display_linked_list(self):
        current = self.head
        index: int = 1
        while current:
            print(f'Number{index:>3} | value = {current.value:>3}')
            current = current.next
            index += 1


if __name__ == "__main__":
    linked_list01 = LinkedList()
    linked_list01.initialize_linked_list()
    linked_list01.display_linked_list()
    print(linked_list01.what_the_fuck)

    # 修改类变量
    LinkedList.what_the_fuck = 2

    linked_list02 = LinkedList()
    linked_list02.initialize_linked_list()
    linked_list02.display_linked_list()
    print(linked_list02.what_the_fuck)
