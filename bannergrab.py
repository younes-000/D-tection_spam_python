import socket



def banner(ip,port):
    s = socket.socket()
    s.connect((ip,int(port)))


def main():

     ip = input("Entrez votre IP: ")
     port = str(input("Entrez votre port: "))
     banner(ip,port)


main()

