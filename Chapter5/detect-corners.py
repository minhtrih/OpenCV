import cv2
import numpy as np

img = cv2.imread('./images/box.jpg')
height, width = img.shape[:2]
img = cv2.resize(src=img, dsize=(int(width/2), int(height/2)))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gray = np.float32(gray)

######### Harrris corner #########
# # to detect only sharp corners
# dst = cv2.cornerHarris(gray, 4, 5, 0.04)
# # to detect soft corners
# # dst = cv2.cornerHarris(gray, 14, 5, 0.04)
# # Result is dilated for marking the corners
# dst = cv2.dilate(dst, None)
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst > 0.01*dst.max()] = [0, 0, 0]

######### Good Features To Track #########
corners = cv2.goodFeaturesToTrack(gray, 7, 0.05, 25)
corners = np.float32(corners)
for item in corners:
    x, y = item[0]
    cv2.circle(img, (x, y), 5, 255, -1)

cv2.imshow("Top 'k' features", img)
cv2.waitKey()

# cv2.imshow('Harris Corners', img)
# cv2.waitKey()
