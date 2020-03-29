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


def get_grid(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 90, 150, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

    if lines is None:
        return

    if filter:
        rho_threshold = 15
        theta_threshold = 0.1

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
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:

                    line_flags[indices[j]] = False

    filtered_lines = []

    if filter:
        for i in range(len(lines)):
            if line_flags[i]:
                filtered_lines.append(lines[i])

    else:
        filtered_lines = lines

    return filtered_lines


def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return x0, y0


def segmented_intersections(lines):
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


def process(image):

    global grid_mapping

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 90, 150, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 275,
                            minLineLength=600, maxLineGap=100)[0].tolist()

    print(lines)

    for x1, y1, x2, y2 in lines:
        for index, (x3, y3, x4, y4) in enumerate(lines):

            if y1 == y2 and y3 == y4:
                diff = abs(y1-y3)
            elif x1 == x2 and x3 == x4:
                diff = abs(x1-x3)
            else:
                diff = 0

            if diff < 10 and diff is not 0:
                del lines[index]

    gridsize = (len(lines) - 2) / q2
    print(gridsize)
    return image


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
