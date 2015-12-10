__author__ = 'JEFFERYK'
import re

pattern = re.compile(r"^Hi [a-zA-Z]*$")
instr = input()
while instr != "BYE":
    if re.match(pattern,instr):
        myname = re.sub(r"^Hi ", "", instr)
        print("Hello, ", myname)
    else:
        print("???")
    instr = input()
