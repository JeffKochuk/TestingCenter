__author__ = 'JEFFERYK'

def sort(ar):
    for i in range(len(ar)):
        smallest = i
        for j in range(i,len(ar)):
            if ar[smallest] < ar[j]:
                smallest = j
        swap(ar,i,smallest)

    return ar

def swap(ar, i1, i2):
    temp = ar[i1]
    ar[i1]=ar[i2]
    ar[i2]=temp
    return ar

inputval = int(input("Enter a value, -1 to quit: "))
vals = []
while inputval != -1:
    vals.append(inputval)
    inputval = int(input("Enter a value, -1 to quit: "))
print(vals)
print(sort(vals))


