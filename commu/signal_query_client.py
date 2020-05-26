#!/usr/bin/python
# -*- coding: UTF-8 -*-
from threading import Timer
import time
import socket
from enum import Enum
import threading
from utils import *


class QueryClientClass(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.threadRunningFlag = 1
        self.name = name
        self.connectState = CliStatEnum.UnConnect
        #self.server_addr = ('37.44.137.30', 8888)
        self.modeType = ModeTypeEnum.PluseMode
        self.serverIP = "127.0.0.1"
        self.serverPort =  8888

        # self.serverIP = "37.44.137.37"
        # self.serverPort =  8888

        self.serverAddr = (self.serverIP, self.serverPort)
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.curTime = time.time()
        
        #self.conQueryTimeOutTimer = threading.Timer(2, self.doConQueryTimeOut, ())
        #self.conQueryCmdNextTimer = threading.Timer(10, self.doConQueryFunc, ())
        self.conQueryNoAckCnt = 0
        self.conQueryAck      = 0
        self.conQueryTime = time.time()
        self.conQueryTimeInterval = 10
        self.conQueryTimeOut      = 2
        
    def __del__(self):
        self.threadRunningFlag = 0
        self.clientSocket.shutdown(socket.SHUT_RDWR)
        self.clientSocket.close()
        self.connectState = CliStatEnum.UnConnect
    
    def resetClient(self):
        try:
            self.clientSocket.shutdown(socket.SHUT_RDWR)
        except Exception as e:
            pass
        self.clientSocket.close()
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connectState = CliStatEnum.UnConnect
        #self.stopSchedConQuery()
        
    def setModeType(self,modeType):
        self.modeType = modeType
        print("setModeType type= {}".format(str(self.modeType)))

    def doReConntion(self):
        while((self.connectState != CliStatEnum.ConnectOk) and (self.connectState != CliStatEnum.InitConnect)):
            self.initConnection()
            time.sleep(5)
            
    def initConnection(self):
        try:
            self.clientSocket.connect(self.serverAddr)
            self.clientSocket.setblocking(False)
        except Exception as err:
            print("initConnection connect error")
            return
        
        if(self.modeType == ModeTypeEnum.PluseMode):
            msg = bytes().fromhex('ff 11 81 01')
        else:
            msg = bytes().fromhex('ff 11 81 02')
        self.connectState = CliStatEnum.InitConnect
        self.clientSocket.send(msg)
        print("initConnection type= {}".format(str(self.modeType)))

    def checkConQuery(self):
        if(self.connectState != CliStatEnum.ConnectOk):
            return 1
     
        nowTime = time.time()    
        if(nowTime <(self.conQueryTime + self.conQueryTimeOut)):
            return 1
           
        if(nowTime <(self.conQueryTime + self.conQueryTimeInterval)):
            if(self.conQueryAck):
               self.conQueryNoAckCnt = 0
               return 1
            else:
               self.conQueryNoAckCnt = self.conQueryNoAckCnt + 1
               
            if(self.conQueryNoAckCnt >=3):
                print("conQuery error {} time no ack,connection is off".format(self.conQueryNoAckCnt))
                self.resetClient()
                return 0
            else:
                self.sendConnectionQueryCmd()
                return 1
        else:
            self.sendConnectionQueryCmd()
            return 1   

        
    def handleQueryConAckCmd(self,recvData):
        #print("handleQueryConCmd {}".format(str(recvData)))
        self.conQueryAck = 1
        
    def sendConnectionQueryCmd(self):
        if(self.connectState != CliStatEnum.ConnectOk):
            return
            
        if(self.modeType == ModeTypeEnum.PluseMode):
            msg = bytes().fromhex('ff 11 80 01')
        else:
            msg = bytes().fromhex('ff 11 80 02')
        self.conQueryAck = 0  
        self.conQueryTime = time.time()
        self.clientSocket.send(msg)
    '''    
    def sendQueryCmd(self):
        if(self.modeType != ModeTypeEnum.QueryMode):
            print("sendQueryCmd error type= {}".format(str(self.modeType)))
            return
            
        msg = bytes().fromhex('ff 11 80 02')
        self.clientSocket.send(msg)
        print("sendQueryCmd type= {}".format(str(self.modeType)))
    '''  
    def handleInitAck(self,recvData):
        if(self.connectState != CliStatEnum.InitConnect):
            print("handleInitAck error connectState {}".format(str(self.connectState)))
            return
        self.connectState = CliStatEnum.ConnectOk
    
    def handleQueryData(self,recvData):
        if(self.connectState != CliStatEnum.ConnectOk):
            print("handleQueryData error connectState {}".format(str(self.connectState)))
            return
        print("handleQueryData")
        print(recvData)
        
    def handlePluseData(self,recvData):
        if(self.connectState != CliStatEnum.ConnectOk):
            print("handlePluseData error connectState {}".format(str(self.connectState)))
            return
        
        
        preTime = self.curTime
        self.curTime = time.time()
        print("handlePluseData")
        print('-----------------total time:{:4f}--------------------'.format(self.curTime - preTime))
        print(recvData)
            
    def handleRevc(self,recvData):
        if(recvData[2] == 0x84):
            self.handleInitAck(recvData)
        elif(recvData[2] == 0x82):
            self.handlePluseData(recvData)
        elif(recvData[2] == 0x83):
            self.handleQueryConAckCmd(recvData)
            
    def run(self):
        print ("开始线程：" + self.name)
        while self.threadRunningFlag:
            self.doReConntion()
            if(0 == self.checkConQuery()):
                continue
            try:
                time.sleep(0.05)#sleep 50ms
                recvData = self.clientSocket.recv(1024) # should no wait
                print('recv_data: {}'.format(recvData))
            except BlockingIOError as e:
                    pass
            except Exception as e:
                print('recv data error! : {}'.format(str(e)))
                self.resetClient()
            else:
                # print(current_state)
                if(recvData!=b''):
                    self.handleRevc(recvData)
                else:
                    self.resetClient()
        print ("退出线程：" + self.name)
        
    
def schedQuery(queryCli):
    queryTimer = threading.Timer(10, doQueryFunc, (queryCli,))
    queryTimer.start()

def doQueryFunc(queryCli):
    queryCli.sendQueryCmd()
    schedQuery(queryCli)
    
def startQueryMode():
    queryCli = QueryClientClass(1,"queryCli")
    queryCli.setModeType(ModeTypeEnum.QueryMode)
    queryCli.initConnection()
    queryCli.start()
    schedQuery(queryCli)
    
def startPluseMode():
    pluseCli = QueryClientClass(1,"pluseCli")
    pluseCli.setModeType(ModeTypeEnum.PluseMode)
    pluseCli.initConnection()
    pluseCli.start()
    
    
def main():
    #startQueryMode()
    startPluseMode()
    
if __name__ == "__main__":
    main()
