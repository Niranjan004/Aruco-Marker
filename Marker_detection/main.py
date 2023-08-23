#pip install opencv-contrib-python==4.6.0.66

import cv2 as cv
from cv2 import aruco
import numpy as np

marker_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

param_markers = cv.aruco.DetectorParameters()

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            cv.polylines(frame, [corners.astype(np.int32)], True, (0, 0, 255), 3, cv.LINE_AA)
            corners=corners.reshape(4,2).astype(int)
            top_right = corners[0].ravel()
            cv.putText(frame, f"id: {ids[0]}", top_right, cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA)
            # print(ids,"    ",corners)

    cv.imshow("Frame", frame)

    key=cv.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()