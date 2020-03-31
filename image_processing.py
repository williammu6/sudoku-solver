import cv2
import math
import numpy as np
from scipy import ndimage
from imutils import contours
from model import predict
from collections import defaultdict
from functools import cmp_to_key
import operator
import threading


from sudoku_solver import Solver

cells = {}


def pre_process_image(img):
    proc = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    proc = cv2.adaptiveThreshold(
        proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    proc = cv2.bitwise_not(proc, proc)

    for _ in range(3):
        proc = cv2.erode(proc, (3, 3), -1)
        proc = cv2.dilate(proc, (3, 3), -1)

    return proc



def denoise(img):
    thresh = cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2)

    blur = cv2.GaussianBlur(img, (11, 11), -1)

    thresh_blur = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    out = thresh & thresh_blur

    out = cv2.erode(out, (5, 5), -1)
    out = cv2.dilate(out, (3, 3), -1)

    return out

def get_cells(img):
    cells = {}

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    thresh = denoise(gray)

    mask = thresh.copy()
    mask[:] = 0
    
    cv2.imshow('thresh', thresh)

    lines = cv2.HoughLines(thresh, 1, np.pi/180, 150)

    thresh = thresh
    if lines is None or len(lines) > 3000:
        return img

    rho_threshold = 25
    theta_threshold = 0.5

    similar_lines = {i: [] for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue

            rho_i, theta_i = lines[i][0]
            rho_j, theta_j = lines[j][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                similar_lines[i].append(j)

    indices = [i for i in range(len(lines))]
    indices.sort(key=lambda x: len(similar_lines[x]))

    line_flags = len(lines)*[True]
    for i in range(len(lines) - 1):

        if not line_flags[indices[i]]:
            continue

        for j in range(i + 1, len(lines)):

            if not line_flags[indices[j]]:
                continue

            rho_i, theta_i = lines[indices[i]][0]
            rho_j, theta_j = lines[indices[j]][0]
            try:
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:

                    line_flags[indices[j]] = False
            except:
                pass

    filtered_lines = []

    for i in range(len(lines)):
        if line_flags[i]:
            filtered_lines.append(lines[i])

    real_lines = []

    for line in filtered_lines:

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

        if abs(angle) < 5 or abs(angle) > 45:
            real_lines.append(line)
            cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 5)

    mask = cv2.erode(mask, (3, 3), -1)

    cv2.imshow('mask', mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    z = 1

    cIndex = 0
    
    # contours = contours[:81]

    for c in contours:
        area = cv2.contourArea(c)
        
        (x, y, w, h) = cv2.boundingRect(c)

        if area > 200 and area < 2000 and abs(w-h) < 10:
            
            cell = thresh[y:y+h, x:x+w]

            cells[cIndex] = {
                'point': (x, y)
            }

            color = (0, 255, 0)

            if np.mean(cell) > 100:
                # cells[cIndex]['prediction'] = predict(cell)
                color = (255, 0, 255)
    
            cv2.rectangle(img, (x+z, y+z), (x+w-z, y+h-z), color, 2)
            # number = predict(cell)
            # cells[cIndex]['prediction'] = number

            cIndex += 1

    return img, cells


def get_roi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

    z = 10
    if best_cnt.any():
        cv2.drawContours(mask, [best_cnt], 0, 255, -1)
        cv2.drawContours(mask, [best_cnt], 0, 0, 2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 1:
            area = cv2.contourArea(contours[0])
            (x, y, w, h) = cv2.boundingRect(contours[0])
            roi = img[y-z:y+h+z, x-z:x+w+z]
            if area > 80000 and roi.any():
                return roi, True
    return img, False


def adjust_rotation(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

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
