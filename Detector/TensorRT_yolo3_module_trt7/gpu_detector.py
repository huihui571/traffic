import sys, os
import time
import random
import numpy as np
import cv2
import torch
import pickle as pkl
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from Detector.TensorRT_yolo3_module_trt7.util import *
import Detector.TensorRT_yolo3_module_trt7.common as common

TRT_LOGGER = trt.Logger()


def get_engine(engine_file_path):
    if os.path.exists(engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print("TRT file not found")


def prep_image(orig_im, inp_dim):
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()  # (3 608 608)
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    img_ = img_.numpy()
    return img_, orig_im, dim


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    return canvas


class Model():
    def __init__(self):
        a = torch.cuda.FloatTensor()  # pytorch必须首先占用部分CUDA
        builder = trt.Builder(TRT_LOGGER)
        # builder.fp16_mode = True
        # builder.strict_type_constraints = True
        self.trt_file = "./Detector/TensorRT_yolo3_module_trt7/convert/yolov3.trt"
        self.use_cuda = True if torch.cuda.is_available() else False
        self.inp_dim = 416
        self.num_classes = 80
        self.classes = load_classes("./Detector/TensorRT_yolo3_module_trt7/cfg/coco.names")
        self.colors = pkl.load(open("./Detector/TensorRT_yolo3_module_trt7/cfg/pallete", "rb"))
        self.output_shapes = [(1, 255, 13, 13), (1, 255, 26, 26), (1, 255, 52, 52)]
        self.yolo_anchors = [[(116, 90), (156, 198), (373, 326)],
                             [(30, 61), (62, 45), (59, 119)],
                             [(10, 13), (16, 30), (33, 23)]]

        self.engine = get_engine(self.trt_file)
        self.inputs_1, self.outputs_1, self.bindings_1, self.stream_1 = common.allocate_buffers(self.engine)
        self.inputs_2, self.outputs_2, self.bindings_2, self.stream_2 = common.allocate_buffers(self.engine)
        self.inputs_3, self.outputs_3, self.bindings_3, self.stream_3 = common.allocate_buffers(self.engine)
        self.inputs_4, self.outputs_4, self.bindings_4, self.stream_4 = common.allocate_buffers(self.engine)
        self.inputs_list = [self.inputs_1, self.inputs_2, self.inputs_3, self.inputs_4]
        self.outputs_list = [self.outputs_1, self.outputs_2, self.outputs_3, self.outputs_4]
        self.bindings_list = [self.bindings_1, self.bindings_2, self.bindings_3, self.bindings_4]
        self.stream_list = [self.stream_1, self.stream_2, self.stream_3, self.stream_4]
        self.context_1 = self.engine.create_execution_context()
        self.context_2 = self.engine.create_execution_context()
        self.context_3 = self.engine.create_execution_context()
        self.context_4 = self.engine.create_execution_context()
        self.context_list = [self.context_1, self.context_2, self.context_3, self.context_4]

    def preparing(self, orig_img_list):
        img = []
        orig_img = []
        im_name = []
        im_dim_list = []
        batch = 1
        for im in orig_img_list:
            im_name_k = ''
            img_k, orig_img_k, im_dim_list_k = prep_image(im, self.inp_dim)
            img.append(img_k)
            orig_img.append(orig_img_k)
            im_name.append(im_name_k)
            im_dim_list.append(im_dim_list_k)

        with torch.no_grad():
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
            im_dim_list_ = im_dim_list

        procession_tuple = (img, orig_img, im_name, im_dim_list)
        return procession_tuple

    def write(self, x, img):
        c1 = tuple(x[1:3].int())  # x1,y1
        c2 = tuple(x[3:5].int())  # x2,y2
        cls = int(x[-1])
        # print("cls:", cls)
        # assert (cls>=0 and cls<self.num_classes)
        if cls < 0 or cls > 80:
            return img
        label = "{0}".format(self.classes[cls])  # class name
        color = random.choice(self.colors)
        if label == "car" or label == "bus" or label == "truck":
            cv2.rectangle(img, c1, c2, (255, 0, 0), 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2, color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                        cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img

    def predict(self, raw_image, cam_id):
        img_list = []
        img_list.append(raw_image)
        procession_tuple = self.preparing(img_list)
        (img, orig_img, im_name, im_dim_list) = procession_tuple

        # with self.engine.create_execution_context() as context:
        if 1:
            context = self.context_list[cam_id]
            inputs, outputs, bindings, stream = self.inputs_list[cam_id], self.outputs_list[cam_id], \
                                                self.bindings_list[cam_id], self.stream_list[cam_id]
            inference_start = time.time()
            inputs[0].host = img[0]  # waiting fix bug
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs,
                                                 stream=stream)
            inference_end = time.time()
            # print('inference time : %f' % (inference_end-inference_start))
            write = 0
            for output, shape, anchors in zip(trt_outputs, self.output_shapes, self.yolo_anchors):
                output = output.reshape(shape)
                trt_output = torch.from_numpy(output).cuda().data
                # trt_output = trt_output.data
                # cuda_time1 = time.time()
                trt_output = predict_transform(trt_output, self.inp_dim, anchors, self.num_classes, self.use_cuda)
                # cuda_time2 = time.time()
                # print('CUDA time : %f' % (cuda_time2 - cuda_time1))
                if type(trt_output) == int:
                    continue

                if not write:
                    detections = trt_output
                    write = 1
                else:
                    detections = torch.cat((detections, trt_output), 1)

            o_time1 = time.time()
            # print('TensorRT inference time : %f' % (o_time1 - inference_start))
            dets = dynamic_write_results(detections, 0.5, self.num_classes, nms=True, nms_conf=0.45)
            o_time2 = time.time()
            # print('After process time : %f' % (o_time2 - o_time1))

            shapes = []
            if not isinstance(dets, int):
                dets = dets.cpu()
                im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
                scaling_factor = torch.min(self.inp_dim / im_dim_list, 1)[0].view(-1, 1)
                dets[:, [1, 3]] -= (self.inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (self.inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])

                list(map(lambda x: self.write(x, orig_img[0]), dets))

                # return detect results in uniform interface

                for x in dets:
                    cls = int(x[-1])
                    score = float(x[-2])
                    conf = float(x[-3])
                    if cls < 0 or cls > 80:
                        label = ""
                    else:
                        label = "{0}".format(self.classes[cls]).replace(" ", "_")  # class name
                    points = list(map(lambda x: int(x.cpu().numpy().tolist()), x[1:5]))
                    shape = {"points": points, "label": label, "score": score}
                    shapes.append(shape)

        return orig_img[0], shapes
