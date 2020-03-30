import cv2

from sudoku_solver import Solver
from image_processing import *

cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # obter ROI contendo SUDOKU
    roi, roi_found = get_roi(gray)

    if roi_found:
        # Rotationa a imagem baseado no ângulo das linhas
        rotated = adjust_rotation(roi)

        # Busca novo ROI removendo as bordas contidas na rotação

        roi, found_roi = get_roi(rotated)

        """
        Obter o sudoku
        """

    cv2.imshow("roi", roi)

    if cv2.waitKey(33) == ord('a'):
        cv2.imwrite('cam_input.png', img)

cv2.destroyAllWindows()
