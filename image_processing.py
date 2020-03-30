import cv2
import math
import numpy as np
from scipy import ndimage
from imutils import contours
from model import predict

from sudoku_solver import Solver

grid_mapping = {}


BLOCK_SIZE = 50
THRESHOLD = 25


def get_roi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    roi = img.copy()
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
            roi = img[y-5:y+h+5, x-5:x+w+5]
            return roi, True
    return img, False


def adjust_rotation(gray_img):

    edges = cv2.Canny(gray_img, 90, 150, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

    angle = 0
    angles = []

    if lines is None:
        return None

    for line in lines:

        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        if abs(angle) < 45:
            angles.append(angle)

    final_angle = np.mean(angles)
    if not math.isnan(final_angle):
        img = ndimage.rotate(img, final_angle)

    return img


def plot_solution(image, solution):
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = cv2.FONT_HERSHEY_SIMPLEX
    THICKNESS = 2

    for i in range(81):
        cell = grid_mapping.get(i, None)
        if cell is None:
            continue

        text = str(solution[i//9][i % 9])
        x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
        textsize = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
        position = (x, y+20)
        cv2.putText(image, text, position, FONT,
                    0.7, (255, 0, 0), THICKNESS)
    return image
