import socket
import threading
def serverThread(lp):
    mySock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM).bind(("localhost",lp))
    while True:
        data, addr = mySock.recvfrom(1024)
        print(">",str(data,"UTF-8"))
def clientThread(sp):
    mySock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        mySock.sendto(bytes(input(), "UTF-8"),("localhost",sp))
thrserv = threading.Thread(target=serverThread,args=(int(input("Listen Port:")),)).start()
thrclient = threading.Thread(target=clientThread,args=(int(input("Send Port:")),)).start()