from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from Detector.YOLO.util import *
from Detector.YOLO.darknet import Darknet
from Detector.YOLO.preprocess import prep_frame, inp_to_image, letterbox_image
import pandas as pd
import random
import pickle as pkl
import argparse
from settings import load_roi, load_stop_line
# import os


class Model():
    def __init__(self):
        self.confidence = 0.5
        self.nms_thresh = 0.4
        self.cfgfile = './Detector/YOLO/cfg/yolov3.cfg'
        self.weightsfile = './Detector/YOLO/yolov3.weights'
        self.reso = '416'
        self.num_classes = 80
        self.CUDA = torch.cuda.is_available()
    
        # print("Path at terminal when executing this file")
        # print(os.getcwd() + "\n")

        self.classes = load_classes('./Detector/YOLO/data/coco.names')
        self.colors = pkl.load(open("./Detector/YOLO/pallete", "rb"))

        print("pre-trained model loading ...")
        self.model = Darknet(self.cfgfile)
        self.model.load_weights(self.weightsfile)
        print("model loaded successfully")

        self.model.net_info["height"] = self.reso
        self.inp_dim = int(self.model.net_info["height"])
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32

        if self.CUDA:
            print('running on GPU')
            self.model.cuda()

  

    def write(self, x, img):
        c1 = tuple(x[1:3].int()) # x1,y1
        c2 = tuple(x[3:5].int()) # x2,y2
        cls = int(x[-1])
        # print("cls:", cls)
        assert (cls>=0 and cls<self.num_classes)
        label = "{0}".format(self.classes[cls]) # class name
        color = random.choice(self.colors)

        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img


    def predict(self, frame):
        inp_dim = self.inp_dim
        CUDA = self.CUDA
        model = self.model

        img, orig_im, dim = prep_frame(frame, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1, 2) #[[2048., 1536., 2048., 1536.]] FIXME:why repeat?
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        # B x (all the boxes) x bbox, torch.Size([1, 10647, 85])
        with torch.no_grad():
            output = model(Variable(img), CUDA)
            # print(output.size())
        output = write_results(output, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thresh)
        # print(output.size())

        im_dim = im_dim.repeat(output.size(0), 1)
        # print(im_dim.size())
        # aa = inp_dim/im_dim
        # bb = torch.min(aa, 1)
        # scaling_factor = bb[0].view(-1, 1) # FIXME:do not need [0]?
        scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)
        # print('scaling_factor', scaling_factor)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1))/2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1))/2
        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        # print(output)
        list(map(lambda x: self.write(x, orig_im), output))

        # return detect results in uniform interface
        shapes = []
        for x in output:
            cls = int(x[-1])
            score = float(x[-2])
            conf = float(x[-3])
            label = "{0}".format(self.classes[cls]).replace(" ", "_")  # class name
            if CUDA:
                points = list(map(lambda x: int(x.cpu().numpy().tolist()), x[1:5]))
            else:
                points = list(map(lambda x: int(x.numpy().tolist()), x[1:5]))
            shape = {"points": points, "label": label, "score": score}
            shapes.append(shape)

        return orig_im, shapes

