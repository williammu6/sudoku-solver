import cv2
import numpy as np
from imutils import contours
from model import predict

from sudoku_solver import Solver

grid_mapping = {}


BLOCK_SIZE = 50
THRESHOLD = 25


def get_grid(img):

    # TODO : Use hough lines to create mask and remove
    #        lines of thresh image before getting numbers
    global grid_mapping

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('gray', gray)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)

    y, x, _ = img.shape
    w_cell = x // 9
    h_cell = y // 9
    rect_cells = []

    cv2.imshow('thresh', thresh)
    index = 0

    mask = thresh.copy()
    mask[:] = 255

    for i in range(10):
        cv2.line(mask, (int(i*w_cell), 0), (int(i*w_cell), y), (0, 0, 0), 3)
        cv2.line(mask, (0, int(i*h_cell)), (x, int(i*h_cell)), (0, 0, 0), 3)

    thresh = thresh & mask

    for i in range(10):
        for j in range(10):
            x1, x2 = int(i*w_cell+2), int(i*w_cell-2)+w_cell
            y1, y2 = int(j*h_cell+2), int(j*h_cell-2)+h_cell

            index += 1

            # if np.mean(cell) > 10:
            #     number = predict(cell)

            #     cv2.putText(img, str(number), (x1, y2),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        # xl = 0+int(i*w_cell)
        # y1, y2 = 0, y
    cv2.imshow('mask', thresh)
    return img


def get_sudoku(image):
    sudoku = np.zeros((9, 9), dtype=np.int8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)

    original_thresh = thresh.copy()
    original_thresh = cv2.erode(original_thresh, (3, 3), -1)

    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    mask_numbers = thresh.copy()
    mask_numbers[:] = 0

    for c in cnts:
        area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)
        if area > 80 and area < 1000:
            cv2.rectangle(mask_numbers, (x+5, y+5), (x+w-5, y+h-5),
                          (255, 255, 255), -1)

    cv2.imshow('thresh', thresh)
    cv2.imshow('mask', mask_numbers)

    cnts = cv2.findContours(mask_numbers, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cnts.sort(key=lambda x: get_contour_precedence(x, thresh.shape[1]))

    cnts = cnts[:81]
    for i, c in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        number_image = original_thresh[y:y+h, x:x+w]
        if np.mean(number_image) > 10:
            prediction = predict(number_image)
            sudoku[i//9, i % 9] = prediction
        else:
            grid_mapping[i] = {
                'x': x,
                'y': y,
                'w': w,
                'h': h,
            }

    return sudoku


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
