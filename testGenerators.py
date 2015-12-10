__author__ = 'JEFFERYK'

def fibonaci(num):
    last = 0
    current = 1
    next = 1
    for n in range(num):
        yield current
        next = current + last
        last = current
        current = next
    return

n = int(input("How many?: "))
for i in fibonaci(n):
    print(i)
