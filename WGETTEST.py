
output = "CONCATENATE("
for y in range(2,200,2):
    output += "IF(HEX2DEC(MID(E2,{},2))=0,\"\",CHAR(HEX2DEC(MID(E2,{},2)))),".format(y-1,y-1)
print(output,")")