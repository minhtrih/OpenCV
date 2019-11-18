import cv2
import numpy as np

mouth_cascade = cv2.CascadeClassifier(
    './data/modules/face/data/cascades/haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
    raise IOError('Unable to load the mouth ear cascade classifier xml file')

cap = cv2.VideoCapture(0)
ds_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor,
                       fy=ds_factor, interpolation=cv2.INTER_AREA)
    # img = cv2.imread('2-ceo-ken-ng.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 5)

    for (x, y, w, h) in mouth_rects:
        y = int(y - 0.15*h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        break

    cv2.imshow('Ear Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
