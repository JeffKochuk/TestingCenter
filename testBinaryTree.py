from BinaryTree import BinTreeNode

inputval = int(input("Enter a value, -1 to quit: "))
head = None
while inputval != -1:
    if head is None:
        head = BinTreeNode(inputval)
    else:
        head.add(BinTreeNode(inputval))
    inputval = int(input("Enter a value, -1 to quit: "))

head.print_tree()
arr = head.toArray()
odds = [n*10 for n in arr if n % 2 == 1]
evens = [n*10 for n in arr if n%2==0]
print(arr)
print(odds)
print(evens)
