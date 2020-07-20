import time
import multiprocessing as mp
import cv2
import gc
import os
import numpy as np
import logging
import sys
import threading

logging.basicConfig(level=logging.WARNING)
show_flag = True

from settings import cam_addrs, roi, stop_line, crop_offset, crop_size, show_img_size
# from Detector.YOLO.gpu_detector import Model
# from Detector.EfficientDet.gpu_detector import Model
# from Detector.EfficientDet.cpu_detector import Model
from Detector.TensorRT_yolo3_module_trt7.gpu_detector import Model
from count import get_car_num, draw_counts
from commu.detect_server import run_detect_server


def crop_image(src_img, cam_id):
    det_img = src_img[crop_offset[cam_id][0]: crop_offset[cam_id][0] + crop_size[0],
              crop_offset[cam_id][1] : crop_offset[cam_id][1] + crop_size[1]]

    return det_img


def roi_init(cam_addrs):
    '''should be executed only once'''
    for cam_id in range(len(cam_addrs)):
        for ch in range(len(roi[cam_id])):
            for p in range(len(roi[cam_id][ch])):
                    roi[cam_id][ch][p][0] = roi[cam_id][ch][p][0] - crop_offset[cam_id][1]
                    roi[cam_id][ch][p][1] = roi[cam_id][ch][p][1] - crop_offset[cam_id][0]

        for line in range(len(stop_line[cam_id])):
            stop_line[cam_id][line] = stop_line[cam_id][line] - crop_offset[cam_id][0]
        print("cam_id:{}, roi:{}".format(cam_id, roi[cam_id]))


def push_image(raw_q, cam_addr, cam_id=0):
    cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
    once = True
    while True:
        is_opened, frame = cap.read()
        if is_opened:
            if once:
                logging.info("write cam{} original image, size:{}".format(cam_id, frame.shape))
                once = False
            raw_q.put(frame)
        else:
            err_pid = os.getpid()
            logging.warning('{} reconnecting ...'.format(err_pid))
            cap = cv2.VideoCapture(cam_addr, cv2.CAP_FFMPEG)
            is_opened, frame = cap.read()
            if is_opened:
                raw_q.put(frame)
        while raw_q.qsize() > 1:
            raw_q.get()
            # gc.collect()
        else:
            time.sleep(0.01)


# def predict_in_thread(model, raw_q, cam_id, tcp_q, show_q):
def predict_in_thread(model, raw_q, cam_id, show_q):
    is_opened = True
    MAX_SIZE = 3
    frame_index = 0
    car_num_q = np.zeros((MAX_SIZE, 2, 3))

    while is_opened:
        logging.info('thread {} blocked: raw_q: {}, pred_q: {}'.format(cam_id , raw_q.qsize(), show_q.qsize()))
        raw_img = raw_q.get()
        logging.info('thread {} got image: raw_q: {}, pred_q: {}'.format(cam_id, raw_q.qsize(), show_q.qsize()))

        raw_img= crop_image(raw_img, cam_id)
        pred_img, pred_result = model.predict(raw_img, cam_id)
        car_num = get_car_num(pred_img, pred_result, roi[cam_id], stop_line[cam_id], (car_num_q, frame_index, MAX_SIZE),
                              smooth="max")
        # gantian
        # pred_img = raw_img
        # car_num = [[0, 0, 0], [0, 0, 0]]

        result = (pred_img, car_num)
        if show_q is not None:
            show_q.put(result)
        # if tcp_q is not None:
        #     tcp_q.put(car_num)
        # while tcp_q.qsize() > 4:
        #     tcp_q.get()


# def predict(raw_qs, tcp_qs=None, show_qs=None):
def predict(raw_qs, show_qs=None):
    roi_init(cam_addrs)
    model = Model()

    for id in range(len(raw_qs)):
        # t = threading.Thread(target=predict_in_thread, args=(model, raw_qs[id], id, tcp_qs[id], show_qs[id], ))
        t = threading.Thread(target=predict_in_thread, args=(model, raw_qs[id], id, show_qs[id], ))
        t.start()


def pop_image(pred_q, window_name, img_shape):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame, car_num = pred_q.get()
        frame = draw_counts(frame, car_num, img_shape)
        # frame = cv2.resize(frame, img_shape)
        # print(car_num)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


def run_single_camera(cam_addr, window_name, img_shape, cam_id=0):
    raw_q = mp.Queue(maxsize=3)
    pred_q = mp.Queue(maxsize=6)

    processes = [
        mp.Process(target=push_image, args=(raw_q, cam_addr, cam_id)),
        mp.Process(target=predict, args=(raw_q, pred_q, cam_id)),
        mp.Process(target=pop_image, args=(pred_q, window_name, img_shape)),
    ]

    [setattr(process, "daemon", True) for process in processes]
    [process.start() for process in processes]
    [process.join() for process in processes]


def run_multi_camera(cam_addrs, window_names, img_shape):
    raw_queues = [mp.Queue(maxsize=3) for _ in cam_addrs]
    pred_queues = [mp.Queue(maxsize=6) for _ in cam_addrs]

    processes = []
    for raw_q, pred_q, cam_addr, window_name in zip(raw_queues, pred_queues, cam_addrs, window_names):
        processes.append(mp.Process(target=push_image, args=(raw_q, cam_addr)))
        processes.append(mp.Process(target=predict, args=(raw_q, pred_q)))
        processes.append(mp.Process(target=pop_image, args=(pred_q, window_name, img_shape)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


def combine_images(img_queue_list, window_name, img_shape):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        t1 = time.time()
        result = [q.get() for q in img_queue_list]
        logging.info('combine got result')

        ###gantian
        imgs = list(map(lambda x: draw_counts(x[0], x[1], img_shape), result))
        # imgs = list(map(lambda x: cv2.resize(x[0], img_shape), result))
        x = np.concatenate(imgs[:2], axis=1)
        y = np.concatenate(imgs[-2:], axis=1)
        imgs = np.concatenate([x, y], axis=0)
        cv2.imshow(window_name, imgs)
        ###gantian

        # cv2.imwrite('det/result.jpg', imgs)
        cv2.waitKey(1)
        t2 = time.time()
        # logging.info('---------show---combine time:{:4f}--------------------'.format(t2-t1))
        print('---------show---combine time:{:4f}--------------------'.format(t2-t1))


def run_multi_camera_in_a_window(cam_addrs, img_shape):
    raw_queues = [mp.Queue(maxsize=2) for _ in cam_addrs]
    show_queues = [mp.Queue(maxsize=4) for _ in cam_addrs]
    tcp_queues = [mp.Queue(maxsize=4) for _ in cam_addrs]

    processes = []
    # processes = [mp.Process(name="dsrv",target=run_detect_server, args=(tcp_queues, ))] #FIXME: report error if remove the ","
    if show_flag:
        # processes.append(mp.Process(name="pred", target=predict, args=(raw_queues, tcp_queues, show_queues)))
        processes.append(mp.Process(name="pred", target=predict, args=(raw_queues, show_queues)))
        processes.append(mp.Process(name="comb",target=combine_images, args=(show_queues, 'CAMs', img_shape)))
    # gantian
    for raw_q, show_q, tcp_q, cam_addr, cam_id in zip(raw_queues, show_queues, tcp_queues, cam_addrs, range(len(cam_addrs))):
        processes.append(mp.Process(name="push",target=push_image, args=(raw_q, cam_addr, cam_id)))
        # if show_flag:
        #     processes.append(mp.Process(name="pred",target=predict, args=(raw_q, cam_id, tcp_q, show_q)))
        # else:
        #     processes.append(mp.Process(name="pred",target=predict, args=(raw_q, cam_id, tcp_q)))
    '''
    processes = [mp.Process(target=combine_images, args=(raw_queues, 'CAMs'))]
    for raw_q, pred_q, cam_addr in zip(raw_queues, pred_queues, cam_addrs):
        processes.append(mp.Process(target=push_image, args=(raw_q, cam_addr)))
    '''

    for process in processes:
        process.daemon = True  # setattr(process, 'deamon', True)
        process.start()
        print("pname {} pid {}".format(process.name,process.pid))
    for process in processes:
        process.join()


def run():
    # cam_addrs = list(map(gst_fmt, cam_addrs))

    # cam_addrs = [
    #     'rtmp://58.200.131.2:1935/livetv/hunantv',
    #     'rtmp://58.200.131.2:1935/livetv/gxtv',
    #     'rtmp://58.200.131.2:1935/livetv/gdtv',
    #     'rtmp://58.200.131.2:1935/livetv/dftv'
    # ]

    # run_single_camera(cam_addrs[0], 'Test', show_img_size, cam_id=0)
    # run_multi_camera(cam_addrs, ['Test' for _ in cam_addrs], show_img_size)
    run_multi_camera_in_a_window(cam_addrs, show_img_size)


if __name__ == '__main__':
    mp.set_start_method(method='spawn')
    run()
