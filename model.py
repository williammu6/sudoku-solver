import os
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from skimage import data, color, exposure
import scipy.ndimage
import numpy as np
import cv2
from skimage.feature import hog
import scipy.misc

from generator import TRAINING_PATH
from constants import *

MODEL = './models/knn_model.pkl'


def get_features(image):
    return hog(image, orientations=8, pixels_per_cell=(3, 3), cells_per_block=(4, 4))


def train():
    X = []
    y = []

    for label in range(0, 10):
        dir = TRAINING_PATH + '/' + str(label) + '/'
        for filename in os.listdir(dir):
            image = cv2.imread(dir + filename)
            X.append(get_features(image))
            y.append(label)

    X = np.array(X, 'float64')
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    joblib.dump(knn, MODEL)


def predict(image):
    knn = joblib.load(MODEL)
    image = cv2.resize(image, IMG_SIZE)
    features = get_features(image)
    predict = knn.predict(features.reshape(1, -1))[0]
    proba = knn.predict_proba(features.reshape(1, -1))
    if np.argmax(proba) <= 0.6:
        return 0
    return predict


if __name__ == '__main__':
    train()
