import cv2
import numpy as np

left_ear_cascade = cv2.CascadeClassifier(
    './data/modules/face/data/cascades/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier(
    './data/modules/face/data/cascades/haarcascade_mcs_rightear.xml')

if left_ear_cascade.empty():
    raise IOError('Unable to load the left ear cascade classifier xml file')
if right_ear_cascade.empty():
    raise IOError('Unable to load the right ear cascade classifier xml file')

cap = cv2.VideoCapture(0)
ds_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor,
                       fy=ds_factor, interpolation=cv2.INTER_AREA)
    # img = cv2.imread('2-ceo-ken-ng.jpg')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    left_ear = left_ear_cascade.detectMultiScale(gray, 1.3, 5)
    right_ear = right_ear_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in left_ear:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    for (x, y, w, h) in right_ear:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

    cv2.imshow('Ear Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
