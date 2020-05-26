#!/usr/bin/python
# -*- coding: UTF-8 -*-
from threading import Timer
import time
import socket
from enum import Enum
import threading
import _thread
from commu.utils import *
from queue import Queue

        
class DetectServerClass(threading.Thread):
    def __init__(self, threadID, name, tcp_q_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.threadRunningFlag = 1
        self.name = name
        self.connectState = ConnectStatEnum.UnConnect
        #self.server_addr = ('37.44.137.37', 8888)
        self.serverIP = "127.0.0.1"
        self.serverPort =  8888
        #self.serverIP = "37.44.137.37"
        #self.serverPort =  8888
        self.serverAddr = (self.serverIP, self.serverPort)
        self.modeType = ModeTypeEnum.UndefinedMode
        self.serverSocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.serverSocket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.cliSockets = []

        self.tcp_q_list = tcp_q_list
        self.detectResQue = Queue()
        #self.cond = threading.Condition()
        #self.camSrv    = CameraServerClass("camSrv",self.detectResQue, self.tcp_q_list)
        #self.camSrv.start()
        
        self.repeatNoConQueryCnt = 0
        self.hasConQueryCmd      = 0
        self.conQueryTime = time.time()
        self.conQueryTimeInterval = 14
        
        self.pluseModeThreadFlags = 0

    def resetConnection(self):
        self.connectState = ConnectStatEnum.UnConnect
        try:
            if(len(self.cliSockets)>0):
                self.cliSockets[0].shutdown(socket.SHUT_RDWR)
                self.cliSockets[0].close()
                del self.cliSockets[0]
        except Exception as e:
            pass

        self.cliSockets = []
        self.stopPluseModeThread()
    '''            
    def checkConQuery(self):        
        if(self.connectState != ConnectStatEnum.Connected):
            return 0
        nowTime = time.time()    
        if(nowTime <(self.conQueryTime + self.conQueryTimeInterval)):
            return 0
            
        if(self.hasConQueryCmd):
           self.repeatNoConQueryCnt = 0
           return 0
           
        self.repeatNoConQueryCnt = self.repeatNoConQueryCnt + 1
        
        if(self.repeatNoConQueryCnt >=3):
            print("checkConQuery error {} time no connect query".format(self.repeatNoConQueryCnt))
            self.resetConnection()
            return 1
        return 0
        #self.schedCheckConQuery()
    '''    
    def checkConQuery(self):        
        if(self.connectState != ConnectStatEnum.Connected):
            return 0
        nowTime = time.time()    
        if(nowTime <(self.conQueryTime + self.conQueryTimeInterval)):
            if(self.hasConQueryCmd):
                self.hasConQueryCmd = 0
                self.conQueryTime = time.time()
                return 0
        else:
            if(0 == self.hasConQueryCmd):
                print("checkConQuery more than 10s,has not recv connect query")
                self.resetConnection()
                return 1
            else:
                self.hasConQueryCmd = 0
                self.conQueryTime = time.time()
                return 0
            
    def replyInitConnect(self):
        if(self.modeType == ModeTypeEnum.PluseMode):
            msg = bytes().fromhex('ff 11 84 01')
            self.cliSockets[0].send(msg)
        elif(self.modeType == ModeTypeEnum.QueryMode):
            msg = bytes().fromhex('ff 11 84 02')
            self.cliSockets[0].send(msg)
        else:
            print('replyInitConnect,error modeType {}'.format(str(self.modeType)))
        
        self.conQueryTime = time.time()
        self.hasConQueryCmd      = 0
        
    def handleInitConnect(self,recvData):
        if(recvData[2] != 0x81):
            print('handleInitConnect,error cmd {}'.format(str(recvData[2])))
            return
        if(recvData[3] == 0x1):
            self.connectState = ConnectStatEnum.Connected
            self.modeType = ModeTypeEnum.PluseMode
            self.startPluseModeThread()
        elif(recvData[3] == 0x2):
            self.connectState = ConnectStatEnum.Connected
            self.modeType = ModeTypeEnum.QueryMode
        else:
            print("handleInitConnect error mode {}".format(str(recvData[3])))
        self.replyInitConnect()

    def handleConQuery(self,recvData):
        self.hasConQueryCmd = 1
        #self.conQueryTime = time.time()
        if(self.modeType == ModeTypeEnum.PluseMode):
            msg = bytes().fromhex('ff 11 83 01')
            self.cliSockets[0].send(msg)
        elif(self.modeType == ModeTypeEnum.QueryMode):
            msg = bytes().fromhex('ff 11 83 02')
            self.cliSockets[0].send(msg)
        else:
            print('handleConQuery,error modeType {}'.format(str(self.modeType)))
        
    def handlePluseModeAck(self,recvData):
        if(self.modeType != ModeTypeEnum.PluseMode):
            return
        print("handlePluseModeAck data {}".format(recvData[3]))
        
    def handleConnected(self,recvData):
        if(self.connectState != ConnectStatEnum.Connected):
            return
        if(recvData[2] == 0x80):
            self.handleConQuery(recvData)
        elif(recvData[2] == 0x85):
            self.handlePluseModeAck(recvData)
        #if(self.modeType != ModeTypeEnum.QueryMode):
        #    return
            
        #if(recvData[3] != 0x02):
        #    return
        #do car detect and send data
        #print("start do car detect");
        #self.camSrv.sendMsg2Camera("cmdDetect");
    
    def handleConnectState(self,connState,recvData):
        if ConnectStatEnum.UnConnect == connState:
            print('ConnectStatEnum.UnConnect')
            self.handleInitConnect(recvData)
        elif ConnectStatEnum.Connected == connState:
            #print('ConnectStatEnum.Connected ')
            self.handleConnected(recvData)
        else:
            print('handleConnectState error')
        
    def run(self):
        print ("start thread" + self.name)
        #try:
        #   _thread.start_new_thread(self.handleDetectResultThread,())
        #except:
        #   print ("Error: handleDetectResultThread")             
        
        self.serverSocket.bind((self.serverIP,self.serverPort)) #绑定要监听的端口
        self.serverSocket.listen(5) #开始监听 表示可以使用1个链接排队
        while self.threadRunningFlag:
            conn,addr = self.serverSocket.accept() #等待链接,多个链接的时候就会出现问题,其实返回了两个值
            print(conn,addr)
            conn.setblocking(False)
            self.cliSockets.append(conn)
            while ((self.threadRunningFlag) and(len(self.cliSockets) > 0)):
                if(self.checkConQuery()):
                    continue
                try:
                    time.sleep(0.05) #sleep 50ms
                    recvData = conn.recv(1024) # should no wait
                    # print('recv_data: {}'.format(recvData))
                except BlockingIOError as e:
                    pass
                except Exception as e:
                    print('recv data error! : {}'.format(str(e)))
                    self.resetConnection()
                else:
                    # print(current_state)
                    if(recvData!=b''):
                        if(len(recvData)>=4):
                            self.handleConnectState(self.connectState,recvData)
                        else:
                            print("recv bad command {}".format(recvData))
                    else:
                        print('!!!recv data empty, reset the connection!')
                        self.resetConnection()
            #conn.close()
        print ("退出线程：" + self.name)


    def handleDetectResultThread(self):
        while self.threadRunningFlag:
            startTime = time.time()
            #print('-----------------handleDetectResultThread:startTime {}--------------------'.format(startTime))
            if((self.detectResQue.qsize()<=0) and (self.threadRunningFlag)):
                time.sleep(0.1) #100ms
                continue
            if(0 == self.threadRunningFlag):
                break
                
            detectRes = self.detectResQue.get(timeout=2.0)
            print("States {}".format(str(self.connectState)))
            print("handleDetectResultThread {}".format(str(detectRes)))
            #TODO: 发送到客户端

            # decouple the send order and the predict order
            num = []
            num.append(detectRes[2][1])  # north straight
            num.append(detectRes[2][0])  # ''    left
            num.append(detectRes[3][1])  # east straight
            num.append(detectRes[3][0])  # ''    left
            num.append(detectRes[0][1])  # south straight
            num.append(detectRes[0][0])  # ''    left
            num.append(detectRes[1][1])  # west straight
            num.append(detectRes[1][0])  # ''    left

            msg = []
            head = [0xff, 0x11, 0x82, 0x03]
            msg.extend(head)
            for i in range(8):
                # if num[i][0] == 0 and num[i][1] == 0 and num[i][2] == 0:
                #     num[i] = [0xee, 0xee, 0xee]
                msg.extend(num[i])
            
            # msg = bytes().fromhex('ff 11 83 01 01') # 这是连接应答吗？不是在这里吧
            if(len(self.cliSockets)>0):
                self.cliSockets[0].send(bytes(msg))
            endTime = time.time()
            #print('-----------------handleDetectResultThread:endTime {}--------------------'.format(endTime))

            # msg = bytes().fromhex('ff 11 83 01 01') # 这是连接应答吗？不是在这里吧
            # self.cliSockets[0].send(msg)

    def startPluseModeThread(self):
        if(self.modeType == ModeTypeEnum.PluseMode):
            self.pluseModeThreadFlags = 1
            try:
               _thread.start_new_thread(self.pluseModeThread,())
            except:
               print ("Error: pluseModeThread")
    def stopPluseModeThread(self):
        if(self.modeType == ModeTypeEnum.PluseMode):
            self.pluseModeThreadFlags = 0
    def pluseModeThread(self):
        print ("start pluseModeThread ")
        while self.pluseModeThreadFlags:
            #self.camSrv.sendMsg2Camera("cmdDetect");
            print("before get tcp_q_list")
            car_num = [q.get() for q in self.tcp_q_list]
            print("after get tcp_q_list")
            num = []
            num.append(car_num[2][1])  # north straight
            num.append(car_num[2][0])  # ''    left
            num.append(car_num[3][1])  # east straight
            num.append(car_num[3][0])  # ''    left
            num.append(car_num[0][1])  # south straight
            num.append(car_num[0][0])  # ''    left
            num.append(car_num[1][1])  # west straight
            num.append(car_num[1][0])  # ''    left

            msg = []
            head = [0xff, 0x11, 0x82, 0x03]
            msg.extend(head)
            for i in range(8):
                # if num[i][0] == 0 and num[i][1] == 0 and num[i][2] == 0:
                #     num[i] = [0xee, 0xee, 0xee]
                msg.extend(num[i])
            
            # msg = bytes().fromhex('ff 11 83 01 01') # 这是连接应答吗？不是在这里吧
            if(len(self.cliSockets)>0):
                self.cliSockets[0].send(bytes(msg))
            endTime = time.time()
            time.sleep(1) #10s
            
    def __del__(self):
        self.threadRunningFlag = 0
        self.serverSocket.close()
        #del self.camSrv



def run_detect_server(q_list):
    print ("detect_server")
    detectSrv = DetectServerClass(1,"DetectServer", q_list)
    detectSrv.start()

'''
if __name__ == "__main__":
    main()
'''

