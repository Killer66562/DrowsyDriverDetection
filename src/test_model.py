from keras.api.layers import Resizing, Rescaling
from keras.api.models import Sequential, load_model
from keras.api.utils import load_img
from keras.api.preprocessing.image import load_img, img_to_array

import cv2
import numpy as np


MODEL_PATH = "models/model-20241204-175233.keras"
DROWSY_IMG_PATH = "test_images/v2.jpeg"
NOT_DROWSY_IMG_PATH = "test_images/v2.jpeg"

IMG_HEIGHT = 128
IMG_WIDTH = 128
COLOR_MODE = "rgb"

DIM = 4 if COLOR_MODE == "rgba" else 1 if COLOR_MODE == "grayscale" else 3


model = load_model(MODEL_PATH)

drowsy_img = load_img(DROWSY_IMG_PATH, color_mode=COLOR_MODE, target_size=(IMG_HEIGHT, IMG_WIDTH))
not_drowsy_img = load_img(NOT_DROWSY_IMG_PATH, color_mode=COLOR_MODE, target_size=(IMG_HEIGHT, IMG_WIDTH))

drowsy_arr = img_to_array(drowsy_img, dtype=np.uint8)
not_drowsy_arr = img_to_array(not_drowsy_img, dtype=np.uint8)

drowsy_arr = cv2.cvtColor(drowsy_arr, cv2.COLOR_RGB2BGR)
not_drowsy_arr = cv2.cvtColor(not_drowsy_arr, cv2.COLOR_RGB2BGR)

'''
cv2.imshow("11", drowsy_arr)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

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