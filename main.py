import cv2

from sudoku_solver import Solver
from image_processing import *

cap = cv2.VideoCapture(0)
while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    roi = image.copy()
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    mask = np.zeros((gray.shape), np.uint8)
    best_cnt = np.array(0)

    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = i

    if best_cnt.any():
        cv2.drawContours(mask, [best_cnt], 0, 255, -1)
        cv2.drawContours(mask, [best_cnt], 0, 0, 2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 1:
            area = cv2.contourArea(contours[0])
            (x, y, w, h) = cv2.boundingRect(contours[0])
            roi = image[y:y+h, x:x+w]

            sudoku = get_grid(roi)

    cv2.imshow("roi", roi)

    if cv2.waitKey(33) == ord('a'):
        cv2.imwrite('cam_input.png', image)

# print(sudoku)

cv2.destroyAllWindows()
