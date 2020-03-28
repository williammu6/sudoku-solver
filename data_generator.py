import cv2
import numpy as np
import os
import shutil

from constants import *

FONT_SCALE = [0.8, 0.9, 1, 1.1]
THICKNESS = [1, 2]

fonts = [
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
]

TRAINING_PATH = './dataset/training'

images = []


def generate():
    if os.path.exists('./dataset'):
        shutil.rmtree('./dataset')
    os.mkdir('./dataset')
    os.mkdir(TRAINING_PATH)
    os.mkdir('./dataset/validation')

    id = 0
    for n in range(10):
        dest = f'{TRAINING_PATH}/{n}'
        print(f'Generating {n}')
        os.mkdir(dest)
        for font in fonts:
            for thickness in THICKNESS:
                for font_scale in FONT_SCALE:

                    textsize = cv2.getTextSize(
                        str(n), font, font_scale, thickness)[0]
                    x = (IMG_SIZE[0]-textsize[0])//2
                    y = (IMG_SIZE[1]+textsize[1])//2

                    for nx in [-3, 0, 3]:
                        img = np.zeros(IMG_SIZE)
                        cv2.putText(img, str(n), (x+nx, y),
                                    font, font_scale, (255, 255, 255), thickness)
                        cv2.imwrite(dest + '/' + str(id) + '.png', img)
                        id += 1


if __name__ == '__main__':
    generate()
