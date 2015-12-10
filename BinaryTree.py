__author__ = 'JEFFERYK'

class BinTreeNode:
    left = None
    right = None
    val = None
    def __init__(self, x):
        self.val = x
    #new is BinTreeNode
    def add(self, new):
        if new.val < self.val:
            if self.left is None:
                self.left = new
            else:
                self.left.add(new)
        else:
            if self.right is None:
                self.right = new
            else:
                self.right.add(new)
        return

    def print_tree(self):
        for n in self.traverse():
            print(n)
        return

    def toArray(self):
        arr = []
        for n in self.traverse():
            arr.append(n)
        return arr

    def traverse(self):
        if not self.left is None:
            for n in self.left.traverse():
                yield n
        yield self.val
        if not self.right is None:
            for n in self.right.traverse():
                yield n
        return





