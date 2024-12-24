import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from keras.api.layers import Resizing, Rescaling
from keras.api.models import Sequential, load_model
from keras.api.preprocessing.image import load_img, img_to_array

from PIL import Image


MODEL_PATH = "prediction_models/model-20241204-191210.keras"
IMG_HEIGHT = 227
IMG_WIDTH = 227
COLOR_MODE = "rgb"

DIM = 4 if COLOR_MODE == "rgba" else 1 if COLOR_MODE == "grayscale" else 3

VIDEO_DEVICE_NUMBER = 1

model = load_model(MODEL_PATH)

normalization = Sequential([
    Resizing(IMG_HEIGHT, IMG_WIDTH), 
    Rescaling(1. / 255)
])

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='models/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.IMAGE)

drowsy_for = 0
drowsy_max = 500
drowsy_times = 0

with FaceDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(VIDEO_DEVICE_NUMBER)
    while True:
        ret, frame = cap.read()             # 讀取影片的每一幀
        w = frame.shape[1]
        h = frame.shape[0]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            lx = bbox.origin_x
            ly = bbox.origin_y
            width = bbox.width
            height = bbox.height
            detected_face = (lx, ly, lx + width, ly + height)
            
            face = Image.fromarray(frame).crop(detected_face)
            face_np = np.asarray(face)

            cv2.rectangle(frame, (lx, ly), (lx + width, ly + height), (0, 0, 255), 5)

            image = tf.constant(face_np, dtype=np.uint8)
            data = [image]

            data = normalization(data)
            result = model.predict(data)
            drowsy = True if result[0][0] > 0.5 else False

            print(f"Drowsy percentage: %.4f %s" % (result[0][0] * 100, "%"))
            print(f"You are %s" % ("drowsy" if drowsy else "not drowsy", ))

            for keyPoint in detection.keypoints:
                #print(keyPoint, w, h)
                cx = int(keyPoint.x * w)
                cy = int(keyPoint.y * h)
                #print(cx, cy)
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
            #print(bbox)
        if not ret:
            print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
            break
        cv2.imshow('Drowsy detection', frame)     # 如果讀取成功，顯示該幀的畫面
        if cv2.waitKey(10) == ord('q'):      # 每一毫秒更新一次，直到按下 q 結束
            break
    cap.release()                           # 所有作業都完成後，釋放資源
    cv2.destroyAllWindows()                 # 結束所有視窗