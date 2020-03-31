import cv2
import numpy as np

from sudoku_solver import Solver
from image_processing import *


def build_sudoku(cells):
    sudoku = np.zeros((9, 9), dtype=np.int8)

    for k in cells.keys():
        if cells[k].get('prediction'):
            sudoku[k // 9][k % 9] = cells[k]['prediction']

    print(sudoku)

def start():
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()

        # obter ROI contendo SUDOKU
        roi, roi_found = get_roi(img)

        if roi.any() and roi_found:
            # Rotaciona a imagem baseado no ângulo das linhas
            rotated = adjust_rotation(roi)
            # Busca novo ROI removendo as bordas contidas na rotação
            roi, found_roi = get_roi(rotated)

            if found_roi:
                # try:
                img, cells = get_cells(roi)
                sudoku = build_sudoku(cells)
                # except Exception as e:
                #     print(e)

                # cv2.imshow('cells', img)


                    
            # except Exception as e:
            #     print(e)
            """
            Obter o sudoku
            """

        cv2.imshow("roi", roi)

        if cv2.waitKey(33) == ord('a'):
            cv2.imwrite('cam_input.png', img)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    start()