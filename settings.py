cam_addrs = [
    "rtsp://admin:admin12345@37.44.169.10/Streaming/Channels/1?tcp",
    "rtsp://admin:admin12345@37.44.169.11/Streaming/Channels/1?tcp",
    "rtsp://admin:admin12345@37.44.169.12/Streaming/Channels/1?tcp",
    "rtsp://admin:admin12345@37.44.169.13/Streaming/Channels/1?tcp"
]

cam_ips = [
    '37.44.169.10',
    '37.44.169.11',
    '37.44.169.12',
    '37.44.169.13',
]
user_name = 'admin'
user_pwd = 'adimn12345'
camera_ip = '37.44.169.10'
show_img_size = (600, 600)

# [left,others]

# roi = [
#     [[[363, 192], [475, 192], [495, 83], [443, 95]], [[103, 194], [362, 192], [487, 35], [106, 41]]],
#     [[[729, 462], [852, 458], [918, 305], [879, 316]], [[403, 477], [724, 463], [922, 274], [400, 299]]],
#     [[[652, 474], [770, 477], [829, 344], [767, 354]], [[335, 473], [646, 474], [831, 286], [336, 294]]],
#     [[[476, 279], [640, 274], [727, 75], [691, 77]], [[163, 287], [474, 280], [687, 78], [164, 87]]]
# ]
# added roi 20200425
roi = [
    [[[349, 208], [463, 209], [491, 52], [474, 55]], [[157, 205], [348, 208], [481, 46], [415, 52]]],
    [[[713, 479], [843, 474], [918, 305], [879, 316]], [[491, 482], [711, 479], [912, 283], [867, 283]]],
    [[[630, 493], [761, 493], [829, 312], [810, 313]], [[421, 493], [630, 493], [830, 291], [791, 292]]],
    [[[461, 296], [619, 296], [727, 75], [691, 77]], [[299, 296], [460, 296], [689, 76], [651, 79]]]
]
# added end
# [ch1, ch2]
'''
stop_line = [
    [208, 118, 78],
    [479, 387, 343],
    [471, 416, 379],
    [276, 215, 175]
]
'''
stop_line = [
    [190, 120, 82],
    [466, 385, 343],
    [493, 388, 345],
    [291, 183, 128]
]
# (y, x)
crop_offset = [[0, 0], [135, 300], [180, 240], [0, 120]]
crop_size = [360, 640]

def gst_fmt(uri, rtsp_latency=200, image_width=416, image_height=416):
    gst_str = ("rtspsrc location={} latency={} ! rtph264depay ! h264parse ! omxh264dec ! nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! videoconvert ! appsink sync=false").format(uri, rtsp_latency, image_width, image_height)
    return gst_str

def load_roi():
    return roi

def load_stop_line():
    return stop_line
