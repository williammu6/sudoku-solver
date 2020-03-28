import cv2
import numpy as np
from imutils import contours
from model import predict

from sudoku_solver import Solver

grid_mapping = {}


def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


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
        if area < 1000:
            cv2.rectangle(mask_numbers, (y+3, x+3), (y+h-3, x+w-3),
                          (255, 255, 255), -1)

    cnts = cv2.findContours(mask_numbers, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cnts.sort(key=lambda x: get_contour_precedence(x, thresh.shape[1]))

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


image = cv2.imread('input.png')
sudoku = get_sudoku(image)


solver = Solver(sudoku)
solver.solve()
solution = solver.sudoku

print()

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

cv2.imshow('solved', image)
cv2.waitKey(0)
