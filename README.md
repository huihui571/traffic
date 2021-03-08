# 十字路口车流统计项目

## 简介

这是一个基于海康威视网络摄像头和英伟达[jetson Tx2](https://www.nvidia.cn/autonomous-machines/embedded-systems/jetson-tx2/)开发板的边缘视觉演示项目。该项目使用开发板连接十字路口的4路摄像头，采用人工智能算法对摄像头采集的实时视频流进行处理，并将处理结果上传给路口信号机和显示终端。

不同于传统的在服务端进行视频处理，该项目提出了一种在边缘端对视频进行处理的方法。相比于传统的集中式处理，该方法节省了网络传输带宽，降低了设备成本和功耗，更满足了对数据处理的实时性和保密性等需求。

## 功能

- [x] 视频拉流
- [x] 目标检测
- [x] 区域车辆统计
- [x] 车辆类型区分
- [x] 结果上传
- [x] 实时显示

## 效果

![效果图](https://ftp.bmp.ovh/imgs/2021/03/0a35878ab85f97c1.jpg)


## 运行
`python3 run.py`
