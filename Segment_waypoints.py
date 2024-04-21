import cv2
import numpy as np

def segmentPipeByBrightness(img):
    # Convert the image to the HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Get the V channel
    v_channel = hsv_img[:, :, 2]

    # Apply thresholding to segment the pipe based on brightness
    thresh, mask = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

cap = cv2.VideoCapture("WhatsApp Video 2024-04-21 at 10.01.17 PM.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        # Apply median filter to reduce noise
        median_filtered = cv2.medianBlur(frame, 3)

        # Segment the pipe based on brightness
        mask = segmentPipeByBrightness(median_filtered)

        # Find edges using Canny edge detector
        canny_edges = cv2.Canny(mask, 100, 200)

        # Find lines using the Hough transform
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

                # Draw the detected lines on the original frame
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Show the segmented mask and the original frame
        cv2.imshow("mask_brightness", mask)
        cv2.imshow("frame", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
