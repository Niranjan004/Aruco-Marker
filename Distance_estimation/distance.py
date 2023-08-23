import cv2 as cv
from cv2 import aruco
import numpy as np

num=input("Enter marker ID: ")

calib_data_path="MultiMatrix.npz"
calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vector = calib_data["rVector"]
t_vector = calib_data["tVector"]

marker_size = 8 # centimeters
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
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners, marker_size, cam_mat, dist_coef)
        total_markers = range(0, marker_IDs.size)

        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(frame, [corners.astype(np.int32)], True, (0, 0, 255), 3, cv.LINE_AA)
            corners=corners.reshape(4,2).astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()

            # Draw pose of the marker
            poit = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 3, 2)

            cv.putText(frame, f"id: {ids[0]} Dist: {round(tVec[i][0][2]), 3}", top_right, cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA)
            cv.putText(frame, f"x: {round(tVec[i][0][0]), 3} y: {round(tVec[i][0][1]), 3}", bottom_right, cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA)
            num=int(num)
            if ids[0]==num:
                print({round(tVec[i][0][0]), 3}, {round(tVec[i][0][1]), 3})

    cv.imshow("Frame", frame)
        

    key=cv.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()