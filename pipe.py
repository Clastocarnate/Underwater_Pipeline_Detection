import cv2
import numpy as np

from sklearn.linear_model import RANSACRegressor

def onlyYellow(img, lower_yellow, upper_yellow):
    b, g, r = cv2.split(img)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(25,25))

    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)

    cl_img = cv2.merge([b, g, r])

    hsv_img = cv2.cvtColor(cl_img, cv2.COLOR_BGR2HSV)

    yellow_hue_range_low = np.array([lower_yellow, 100, 100])
    yellow_hue_range_high = np.array([upper_yellow, 255, 255])
    yellow_mask = cv2.inRange(hsv_img, yellow_hue_range_low, yellow_hue_range_high)
    

    hue_channel = cv2.bitwise_and(hsv_img[:, :, 0], hsv_img[:, :, 0], mask=yellow_mask)
    thresh, mask = cv2.threshold(hue_channel,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

cap = cv2.VideoCapture("1701 12.06.2023 18.mp4")




while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
     
        median_filtered = cv2.medianBlur(frame,3)

        mask = onlyYellow(frame,20,150)

        canny_edges = cv2.Canny(mask,100,200)
        lines = cv2.HoughLines(canny_edges, 1, np.pi / 180, 100)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 2)

        cv2.imshow("mask_yellow",mask)
   
        cv2.imshow("frame",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()