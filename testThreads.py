__author__ = 'JEFFERYK'
import threading
import math

def isprime(x):
    for n in range(2, int(math.sqrt(x))):
        if x % n == 0:
            return False
    return True

def check(start,end):
    for n in range(start, end):
        if isprime(n):
            print(n)

thr1 = threading.Thread(target=check, args=(3,100))
thr2 = threading.Thread(target=check, args=(101,200))
thr1.start()
thr2.start()

