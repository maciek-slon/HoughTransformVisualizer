import sys
import cv2
import numpy as np
import math
import random
from matplotlib import pyplot as plt

WIDTH=400
HEIGHT=400

drawing = False

def convert_acc(acc):
    # tmp = np.sqrt(acc)
    tmp = np.power(acc, 1)
    amax = np.amax(tmp)
    tmp = 255 * tmp / amax
    # return  cv2.applyColorMap(tmp.astype(np.uint8), cv2.COLORMAP_PINK)
    return cv2.cvtColor(tmp.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def get_points(img):
    pts = []
    for y,x in zip(*np.nonzero(img)):
        pts.append([x,y])
    return pts

def redraw(params):
    points = params["points"]
    img = params["img"]
    img_over = params["img_over"]
    acc = params["acc"]
    acc_over = params["acc_over"]
    for pt in points:
        x = pt[0]
        y = pt[1]
        cv2.circle(img, (x,y), 0, (255, 255, 255))
        tmp = draw_sine(np.zeros_like(acc), x, y, 1)
        acc[:,:,0] = cv2.addWeighted(acc, 1, tmp, 1, 0)
    
    img_over[:,:,:] = img
    acc_over[:,:,:] = convert_acc(acc)

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
        acc_over[:,:,:] = draw_sine(acc_over, x, y, (0, 0, 255))
        pass

    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
    
    if drawing:
        points.append([x, y])
        cv2.circle(img, (x,y), 0, (255, 255, 255))
        tmp = draw_sine(np.zeros_like(acc), x, y, 1)
        acc[:,:,0] = cv2.addWeighted(acc, 1, tmp, 1, 0)

    img_over[:,:,:] = img
    
    h, w = img.shape[0:2]
    x = x - w/2
    y = y - h/2

    cv2.putText(img_over, f"{x=:.0f}", (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv2.putText(img_over, f"{y=:.0f}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

def peek_cb(event, x, y, flags, params):
    global drawing

    points = params["points"]
    img = params["img"]
    img_over = params["img_over"]
    acc = params["acc"]
    acc_over = params["acc_over"]

    h, w = acc.shape[0:2]
    astep = math.pi / w

    theta = x * astep
    rho = (y-h/2)*2

    if event==cv2.EVENT_MOUSEMOVE:
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a*rho + w/2
        y0 = b*rho + h/2
        pt1 = ( round(x0 + 1000*(-b)), round(y0 + 1000*(a)) )
        pt2 = ( round(x0 - 1000*(-b)), round(y0 - 1000*(a)) )
        
        ox = round(w/2)
        oy = round(h/2)

        x0 = round(x0)
        y0 = round(y0)

        img_over[:,:,:] = img
        cv2.line(img_over, pt1, pt2, (0, 0, 255))
        cv2.line(img_over, (ox, oy), (x0, y0), (255, 255, 0))
        cv2.ellipse(img_over, (ox, oy), (30, 30), 0, 0, theta * 180 / math.pi, (0, 255, 255))
               

    theta = theta * 180 / math.pi

    acc_over[:,:,:] = convert_acc(acc)
    votes = float(acc[y,x])

    cv2.putText(acc_over, f"{rho=:.1f}", (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv2.putText(acc_over, f"{theta=:.2f}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    cv2.putText(acc_over, f"{votes=:.0f}", (10, 45), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))


def draw_sine(img, x, y, color):
    h, w = img.shape[0:2]
    astep = math.pi / w

    x = x - w/2
    y = y - h/2

    for i in range(w):
        theta = i * astep
        rho = x*math.cos(theta) + y * math.sin(theta)
        cv2.circle(img, (i, round(rho/2 + HEIGHT/2)), 0, color, -1)
        

    return img

def main(argv):
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    acc = np.zeros((HEIGHT, WIDTH, 1), np.float32)

    acc_over = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    img_over = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    cb_params = {
        "points": [],
        "img": img,
        "img_over": img_over,
        "acc": acc,
        "acc_over": acc_over
    }

    redraw(cb_params)
    
    cv2.namedWindow("img")
    cv2.setMouseCallback('img', draw_cb, cb_params)

    cv2.namedWindow("acc")
    cv2.setMouseCallback('acc', peek_cb, cb_params)

    ch = 0
    loading = False
    loaded = 0
    loaded_pts = []
    invert = False

    while ch != 27:

        if ch == ord('c'):
            img[:] = 0
            acc[:] = 0
            img_over[:] = 0
            acc_over[:] = 0
            cb_params['points'] = []
            loading = False

        if ch == ord('1'):
            loaded = 0
            tmp = cv2.imread('sudoku.png', cv2.IMREAD_GRAYSCALE)
            pts = get_points(tmp)
            random.shuffle(pts)
            loaded_pts = pts
            img[:] = cv2.cvtColor(tmp // 4, cv2.COLOR_GRAY2BGR)[:]
            acc[:] = 0
            img_over[:] = 0
            acc_over[:] = 0
            loading = True

        if ch == ord('2'):
            loaded = 0
            tmp = cv2.imread('text.png', cv2.IMREAD_GRAYSCALE)
            pts = get_points(tmp)
            random.shuffle(pts)
            loaded_pts = pts
            img[:] = cv2.cvtColor(tmp // 4, cv2.COLOR_GRAY2BGR)[:]
            acc[:] = 0
            img_over[:] = 0
            acc_over[:] = 0
            loading = True

        if ch == ord(' '):
            loading = not loading

        if ch == ord('h'):
            
            plt.plot(np.max(acc, axis=0))
            plt.show()


        if loading:
            if loaded >= len(loaded_pts):
                loading = False
                cb_params["points"] = loaded_pts
            cb_params["points"] = loaded_pts[loaded:loaded+50]
            loaded = loaded + 50
            redraw(cb_params)
            

        if ch == ord('i'):
            invert = not invert

        if invert:
            acc_show = 255 - acc_over
        else:
            acc_show = acc_over

        cv2.imshow("img", img_over)
        cv2.imshow("acc", acc_show)

        ch = cv2.waitKey(2)


if __name__=="__main__":
    main(sys.argv)