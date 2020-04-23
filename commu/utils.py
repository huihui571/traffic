#!/usr/bin/python
# -*- coding: UTF-8 -*-
import _thread
from enum import Enum


# conn_flag = False
class ConnectStatEnum(Enum):
    UnConnect = 0
    Connected = 1
    ConnectError = 2

class ModeTypeEnum(Enum):
    UndefinedMode = 0
    PluseMode = 1
    QueryMode = 2

    
class CliStatEnum(Enum):
    UnConnect      = 0
    InitConnect    = 1
    ConnectOk      = 2
    QueryMode = 2

