import socket
import threading
def serverThread(lp):
    mySock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mySock.bind(("localhost",lp))
    data, addr = mySock.recvfrom(1024)
    while True:
        print(">",str(data,"UTF-8"))
        data, addr = mySock.recvfrom(1024)
def clientThread(sp):
    mySock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = input()
    while data != "quit":
        mySock.sendto(bytes(data, "UTF-8"),("localhost",sp))
        data = input()
listenPort = int(input("Listen Port:"))
sendPort = int(input("Send Port:"))
thrserv = threading.Thread(target=serverThread,args=(listenPort,)).start()
thrclient = threading.Thread(target=clientThread,args=(sendPort,)).start()
