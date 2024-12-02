from keras.api import Model, Sequential
from keras.api.layers import Input, Dropout, Dense, Conv2D, MaxPool2D, Flatten, \
    Rescaling, RandomFlip, RandomRotation, RandomBrightness
from keras.api.optimizers import Adam, SGD
from keras.api.utils import image_dataset_from_directory
from keras.api.applications import ResNet50V2

from datetime import datetime, timezone, timedelta

import os

SEED = 42

IMG_HEIGHT = 48
IMG_WIDTH = 48

BATCH_SIZE = 200
EPOCHS = 100
LEARNING_RATE = 2.


model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)), 

    Conv2D(4, (3, 3), activation="relu", padding="same"), 
    Conv2D(4, (3, 3), activation="relu", padding="same"), 
    MaxPool2D((2, 2), strides=(3, 3)), 

    Conv2D(8, (3, 3), activation="relu", padding="same"), 
    Conv2D(8, (3, 3), activation="relu", padding="same"), 
    MaxPool2D((2, 2), strides=(3, 3)), 

    Conv2D(16, (3, 3), activation="relu", padding="same"), 
    Conv2D(16, (3, 3), activation="relu", padding="same"), 
    MaxPool2D((2, 2), strides=(3, 3)), 

    Conv2D(32, (3, 3), activation="relu", padding="same"), 
    Conv2D(32, (3, 3), activation="relu", padding="same"), 
    MaxPool2D((2, 2), strides=(3, 3)), 

    Flatten(), 
    Dropout(0.1), 

    Dense(512, activation="relu"), 
    Dense(128, activation='relu'), 
    Dense(32, activation='relu'), 
    Dense(8, activation='relu'), 
    Dense(4, activation='relu'), 
    Dense(2, activation='relu'), 
    Dense(1, activation="sigmoid")
])

model = ResNet50V2(input_shape=(IMG_HEIGHT, IMG_WIDTH))

normalization = Sequential([
    Rescaling(1. / 255)
])

data_augmentation = Sequential([
    RandomFlip("horizontal"), 
    RandomRotation(0.2), 
    RandomBrightness(0.2)
])

optimizer = SGD(learning_rate=LEARNING_RATE)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

datasets_path = os.path.abspath("datasets_new")

datasets_train_path = os.path.join(datasets_path, "train")
datasets_test_path = os.path.join(datasets_path, "test")

dataset_train = image_dataset_from_directory(
    datasets_train_path, 
    color_mode="grayscale", 
    label_mode="binary", 
    image_size=(IMG_HEIGHT, IMG_WIDTH), 
    class_names=["DROWSY", "NATURAL"], 
    interpolation="nearest", 
    batch_size=BATCH_SIZE
)

dataset_validation = image_dataset_from_directory(
    datasets_test_path, 
    color_mode="grayscale", 
    label_mode="binary", 
    image_size=(IMG_HEIGHT, IMG_WIDTH), 
    class_names=["DROWSY", "NATURAL"], 
    interpolation="nearest", 
    batch_size=BATCH_SIZE
)

dataset_train = dataset_train.map(lambda x, y: (data_augmentation(x), y))
dataset_train = dataset_train.map(lambda x, y: (normalization(x), y))

dataset_validation = dataset_validation.map(lambda x, y: (normalization(x), y))

hist = model.fit(
    dataset_train, 
    epochs=EPOCHS, 
    validation_data=dataset_validation, 
)

result = model.evaluate(dataset_validation)
print("accuracy=%.4f" % result[1])

current_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
model.save(f'model-{current_time}.keras')
