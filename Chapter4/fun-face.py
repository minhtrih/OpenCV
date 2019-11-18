import cv2
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier(
    './data/haarcascades/haarcascade_frontalface_alt.xml')
# face_mask = Image.open('./mask_hannibal.png')
face_mask = cv2.imread('./mask_hannibal.png')
# w_mask, h_mask = face_mask.size
# face_mask = (w_mask, h_mask)
h_mask, w_mask = face_mask.shape[:2]

if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
cap = cv2.VideoCapture(0)
scaling_factor = 0.5
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor,
                       fy=scaling_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face_rects:
        if h > 0 and w > 0:
            # Adjust the height and weight parameters depending on the sizes and the locations. You need to play around with these to make sure you get it right.
            h, w = int(0.8*h), int(0.9*w)
            x = int(x + 0.07*w)
            y = int(y + 0.45*h)
            # Extract the region of interest from the image
            frame_roi = frame[y:y+h, x:x+w]
            face_mask_small = cv2.resize(face_mask, (w, h),
                                         interpolation=cv2.INTER_AREA)
            # Convert color image to grayscale and threshold it
            gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(
                gray_mask, 180, 255, cv2.THRESH_BINARY_INV)
            # Create an inverse mask
            mask_inv = cv2.bitwise_not(mask)
            # Use the mask to extract the face mask region of interest
            masked_face = cv2.bitwise_and(
                face_mask_small, face_mask_small, mask=mask)
            try:
                # Use the inverse mask to get the remaining part ofthe image
                masked_frame = cv2.bitwise_and(
                    frame_roi, frame_roi, mask=mask_inv)
                print(masked_frame)
                # add the two images to get the final output
                frame[y:y+h, x:x+w] = cv2.add(masked_face, masked_frame)
            except:
                pass
            # masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
    cv2.imshow('Face Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()
