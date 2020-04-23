import socket
import time
from threading import Timer

s = socket.socket()
s.bind(('localhost', 8888))
s.listen()
print('server listening...')
conn, addr = s.accept()
print('{} is accepted'.format(addr))
# c = socket.socket()
# server_ip = ('37.44.137.37', 6666)
# c.connect(server_ip)
count = 0
conn_flag = False

def send_connect_ask():
    msg = bytes().fromhex('ff 11 81 01')
    conn.send(msg)
    print('send connect ask')
    return

def send_heart_beat():
    # msg = [0xff, 0x11, 0x80, 0x01]
    msg = bytes().fromhex('ff 11 80 01')
    # msg = bytes(msg)
    conn.send(msg)
    print('send heart_beat')
    return

def heart_beat():
    global count
    # print(time.strftime('%Y-%m-%d %H:%M:%S') + 'count: {}'.format(count))
    if not conn_flag:
        send_connect_ask()   # every 5s before connected
    if conn_flag and count == 1:
        send_heart_beat()    # every 10s after connected
    count += 1
    if count == 2:
        count = 0
    Timer(5, heart_beat).start()


heart_beat()

while True:
    recv_data = conn.recv(8)
    print('recv_data: {}'.format(recv_data))
    if recv_data[2] == 0x83:
        print('recv heart beat ack!')
    elif recv_data[2] == 0x84:
        conn_flag = True
        print('connect successful!')






