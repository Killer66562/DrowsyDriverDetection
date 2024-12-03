import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

img = cv2.imread("test_images/v0.jpeg")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(128, 128))

for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Yes", img)
cv2.waitKey(0)