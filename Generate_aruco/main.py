import cv2 as cv
from cv2 import aruco

marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

marker_size=400 # pixels

for id in range(20):

    marker_image = aruco.drawMarker(marker_dict, id, marker_size)
    #cv.imshow("Image", marker_image)
    cv.imwrite(f"Generate_aruco/Marker_Images/Marker_{id}.png", marker_image)
    #cv.waitKey(0)
    #break