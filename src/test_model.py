from keras.api.layers import Resizing, Rescaling
from keras.api.models import Sequential, load_model
from keras.api.utils import load_img
from keras.api.preprocessing.image import load_img, img_to_array

import cv2
import numpy as np
import os


MODELS_ROOT_PATH = "detection_models"
MODEL_TYPE = "eyes"
MODEL_FILENAME = "model-20241211-135617.keras"

MODEL_PATH = os.path.join(MODELS_ROOT_PATH, MODEL_TYPE, MODEL_FILENAME)

IMAGES_ROOT_PATH = "test_images"

DROWSY_IMG_FILENAME = "eyes_open_2.jpg"
NOT_DROWSY_IMG_FILENAME = "eyes_close_1.jpg"

DROWSY_IMG_PATH = os.path.join(IMAGES_ROOT_PATH, DROWSY_IMG_FILENAME)
NOT_DROWSY_IMG_PATH = os.path.join(IMAGES_ROOT_PATH, NOT_DROWSY_IMG_FILENAME)

IMG_HEIGHT = 79
IMG_WIDTH = 79
COLOR_MODE = "grayscale"

DIM = 4 if COLOR_MODE == "rgba" else 1 if COLOR_MODE == "grayscale" else 3


model = load_model(MODEL_PATH)

drowsy_img = load_img(DROWSY_IMG_PATH, color_mode=COLOR_MODE, target_size=(IMG_HEIGHT, IMG_WIDTH))
not_drowsy_img = load_img(NOT_DROWSY_IMG_PATH, color_mode=COLOR_MODE, target_size=(IMG_HEIGHT, IMG_WIDTH))

drowsy_arr = img_to_array(drowsy_img, dtype=np.uint8)
not_drowsy_arr = img_to_array(not_drowsy_img, dtype=np.uint8)

if COLOR_MODE != "grayscale" and COLOR_MODE != "rgba":
    drowsy_arr = cv2.cvtColor(drowsy_arr, cv2.COLOR_RGB2BGR)
    not_drowsy_arr = cv2.cvtColor(not_drowsy_arr, cv2.COLOR_RGB2BGR)


cv2.imshow("11", drowsy_arr)
cv2.waitKey(0)
cv2.destroyAllWindows()


drowsy_arr = drowsy_arr.reshape((IMG_HEIGHT, IMG_WIDTH, DIM))
not_drowsy_arr = not_drowsy_arr.reshape((IMG_HEIGHT, IMG_WIDTH, DIM))

dataset_test = [drowsy_arr, not_drowsy_arr]

normalization = Sequential([
    Resizing(IMG_HEIGHT, IMG_WIDTH), 
    Rescaling(1. / 255)
])

dataset_test = normalization(dataset_test)

result = model.predict(dataset_test)
print(result)