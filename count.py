import numpy as np
import cv2


def isPointInRect(p, rect):
    '''
    judge weather the point is in the area, the rect area must be convex, Counterclockwise
    '''
    a, b, c, d = rect[0], rect[1], rect[2], rect[3]
    t1 = (b[0] - a[0])*(p[1] - a[1]) - (b[1] - a[1])*(p[0] - a[0])
    t2 = (c[0] - b[0])*(p[1] - b[1]) - (c[1] - b[1])*(p[0] - b[0])
    t3 = (d[0] - c[0])*(p[1] - c[1]) - (d[1] - c[1])*(p[0] - c[0])
    t4 = (a[0] - d[0])*(p[1] - d[1]) - (a[1] - d[1])*(p[0] - d[0])

    if (t1>0 and t2>0 and t3>0 and t4>0) or (t1<0 and t2<0 and t3<0 and t4<0):
        return True
    else:
        return False

def draw_stop_line(pred_img, roi, stop_line_y):
    '''
    get the x coordinate of the intersection of the stop_line and the roi area boundary, then draw the line
    '''
    x_1, y_1 = roi[1][3][0], roi[1][3][1]
    x_2, y_2 = roi[1][0][0], roi[1][0][1]
    y = stop_line_y
    st_x_1 = int((x_1 * (y - y_2) / (y_1 - y) + x_2) / (1 + (y - y_2) / (y_1 - y)))
    x_3, y_3 = roi[0][2][0], roi[0][2][1]
    x_4, y_4 = roi[0][1][0], roi[0][1][1]
    st_x_2 = int((x_3 * (y - y_4) / (y_3 - y) + x_4) / (1 + (y - y_4) / (y_3 - y)))

    line_color = (255, 0, 0)
    cv2.line(pred_img, (st_x_1, y), (st_x_2, y), line_color)

def get_car_num(pred_img, pred_result, roi, stop_line):
    '''
    count cars int the roi area for a single img
    '''
    car_num = [[0, 0, 0], [0, 0, 0]]
    pts = np.array(roi)
    line_color = (255, 0, 0)
    # draw roi
    cv2.polylines(pred_img, pts, 1, line_color)
    # draw stop line
    draw_stop_line(pred_img, roi, stop_line[1])
    draw_stop_line(pred_img, roi, stop_line[2])
    for i in range(len(pred_result)):
        x1 = pred_result[i]['points'][0]
        y1 = pred_result[i]['points'][1]
        x2 = pred_result[i]['points'][2]
        y2 = pred_result[i]['points'][3]
        label = pred_result[i]['label']
        score = pred_result[i]['score']

        if not (label == "car" or label == "bus" or label == "truck"):
            continue
            
        center = x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2
        bottom_center = x1 + (x2 - x1) / 2, y2
        for ch in range(len(roi)):
            # judge by the bottom_center of the bounding box
            if isPointInRect(bottom_center, roi[ch]):
                cv2.circle(pred_img, (int(bottom_center[0]), int(bottom_center[1])), 5, (0, 0, 255))
                c_y = center[1]
                if label == "car":
                    if (c_y > stop_line[1]):
                        car_num[ch][0] += 1
                    elif (c_y < stop_line[1]) and (c_y > stop_line[2]):
                        car_num[ch][1] += 1
                    elif (c_y < stop_line[2]):
                        car_num[ch][2] += 1
                elif label == "bus" or label == "truck":
                    # count 3 num for one bus or truck
                    if (c_y > stop_line[1]):
                        car_num[ch][0] += 3
                    elif (c_y < stop_line[1]) and (c_y > stop_line[2]):
                        car_num[ch][1] += 3
                    elif (c_y < stop_line[2]):
                        car_num[ch][2] += 3


    return car_num

def draw_counts(img, car_num, img_shape):
    '''
    pt1 ----
    '''
    img_ = cv2.resize(img, img_shape)
    pt1 = (img_shape[0]-5-60, 20)
    pt2 = (img_shape[0]-5, 20+80)
    line_color = (0, 128, 255)
    cv2.rectangle(img_, pt1, pt2, line_color)
    cv2.line(img_, (pt1[0], pt1[1] + 20), (pt1[0] + 60, pt1[1] + 20), line_color)
    cv2.line(img_, (pt1[0], pt1[1] + 65), (pt1[0] + 60, pt1[1] + 65), line_color)
    cv2.line(img_, (pt1[0] + 30, pt1[1]), (pt1[0] + 30, pt1[1] + 80), line_color)
    cv2.putText(img_, 'left', (pt1[0] + 2, pt1[1] + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color)
    cv2.putText(img_, '{}'.format(car_num[0][0]), (pt1[0] + 5, pt1[1] + 17 +15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color)
    cv2.putText(img_, '{}'.format(car_num[0][1]), (pt1[0] + 5, pt1[1] + 17 +30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color)
    cv2.putText(img_, '{}'.format(car_num[0][2]), (pt1[0] + 5, pt1[1] + 17 +45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color)
    cv2.putText(img_, '{}'.format(sum(car_num[0])), (pt1[0] + 5, pt1[1] + 17 +60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color)
    cv2.putText(img_, 'others', (pt1[0] + 2 + 30, pt1[1] + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color)
    cv2.putText(img_, '{}'.format(car_num[1][0]), (pt1[0] + 5 + 30, pt1[1] + 17 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color)
    cv2.putText(img_, '{}'.format(car_num[1][1]), (pt1[0] + 5 + 30, pt1[1] + 17 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color)
    cv2.putText(img_, '{}'.format(car_num[1][2]), (pt1[0] + 5 + 30, pt1[1] + 17 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color)
    cv2.putText(img_, '{}'.format(sum(car_num[1])), (pt1[0] + 5 + 30, pt1[1] + 17 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, line_color)
    return img_
            

        

