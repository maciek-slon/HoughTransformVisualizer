import sys
import cv2
import numpy as np
import math

WIDTH=400
HEIGHT=400
RADIUS = 20

drawing = False

def convert_acc(acc):
    # tmp = np.sqrt(acc)
    tmp = acc
    amax = np.amax(tmp)
    tmp = 255 * tmp / amax
    # return  cv2.applyColorMap(tmp.astype(np.uint8), cv2.COLORMAP_PINK)
    return cv2.cvtColor(tmp.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def draw_cb(event, x, y, flags, params):
    global drawing

    points = params["points"]
    img = params["img"]
    img_over = params["img_over"]
    acc = params["acc"]
    acc_over = params["acc_over"]

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event==cv2.EVENT_MOUSEMOVE:
        acc_over[:,:,:] = convert_acc(acc)
        acc_over[:,:,:] = draw_circle(acc_over, x, y, (0, 0, 255))
        pass

    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
    
    if drawing:
        points.append([x, y])
        cv2.circle(img, (x,y), 0, (255, 255, 255))
        tmp = draw_circle(np.zeros_like(acc), x, y, 1)
        acc[:,:,0] = cv2.addWeighted(acc, 1, tmp, 1, 0)

    img_over[:,:,:] = img

    cv2.putText(img_over, f"{x=:.0f}", (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv2.putText(img_over, f"{y=:.0f}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

def peek_cb(event, x, y, flags, params):
    global drawing

    points = params["points"]
    img = params["img"]
    img_over = params["img_over"]
    acc = params["acc"]
    acc_over = params["acc_over"]

    if event==cv2.EVENT_MOUSEMOVE:

        img_over[:,:,:] = img
        cv2.circle(img_over, (x, y), RADIUS, (0, 0, 255), 1)

    acc_over[:,:,:] = convert_acc(acc)
    votes = float(acc[y,x])

    cv2.putText(acc_over, f"{x=:.0f}", (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv2.putText(acc_over, f"{y=:.0f}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv2.putText(acc_over, f"{votes=:.0f}", (10, 45), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))


def draw_circle(img, x, y, color):    
    cv2.circle(img, (x, y), RADIUS, color, 1)

    return img

def recalculate(points, params):
    acc = params['acc']
    acc[:,:,:] = np.zeros_like(acc)
    for pt in points:
        x = pt[0]
        y = pt[1]
        tmp = draw_circle(np.zeros_like(acc), x, y, 1)
        acc[:,:,0] = cv2.addWeighted(acc, 1, tmp, 1, 0)

    params['acc_over'][:,:,:] = convert_acc(acc)

def main(argv):
    global RADIUS

    points = []
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    acc = np.zeros((HEIGHT, WIDTH, 1), np.float32)

    acc_over = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    img_over = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    cb_params = {
        "points": points,
        "img": img,
        "img_over": img_over,
        "acc": acc,
        "acc_over": acc_over
    }
    
    cv2.namedWindow("img")
    cv2.setMouseCallback('img', draw_cb, cb_params)

    cv2.namedWindow("acc")
    cv2.setMouseCallback('acc', peek_cb, cb_params)

    ch = -1

    invert = False

    while ch != 27:
        if ch == ord('+'):
            RADIUS = RADIUS + 5
            recalculate(points, cb_params)
        elif ch == ord('-'):
            RADIUS = max(RADIUS-5, 5)
            recalculate(points, cb_params)

        if ch == ord('i'):
            invert = not invert

        if invert:
            acc_show = 255 - acc_over
        else:
            acc_show = acc_over

        cv2.imshow("img", img_over)
        cv2.imshow("acc", acc_show)

        ch = cv2.waitKey(10)


if __name__=="__main__":
    main(sys.argv)