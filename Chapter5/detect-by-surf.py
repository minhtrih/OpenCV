import cv2
import numpy as np

input_image = cv2.imread('./images/car.jpg')
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create(1500)

# This threshold controls the number of keypoints
kp, des = surf.detectAndCompute(gray_image, None)
input_image = cv2.drawKeypoints(input_image, kp, None, (0, 255, 0), 4)


cv2.imshow('SURF features', input_image)
cv2.waitKey()
