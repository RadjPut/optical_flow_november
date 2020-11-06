from OpticalFlow import harris_corners, shi_tomasi, good_features_2_track, optical_flow
import cv2

cap = cv2.VideoCapture(0)

fist_iter = True

#img = cv2.imread('car.png')
#cv2.imshow("Origin img", img)
#cv2.waitKey(9999)
#
#harris_corners = harris_corners(img)
while True:
    ret, fist_frame = cap.read()
#    shi_tomasi_features = shi_tomasi(fist_image)
    if fist_iter:
        p0, old_gray, mask = good_features_2_track(fist_frame)
        fist_iter = False

    #cv2.imshow("optical flow frame", fist_frame)
    #cv2.waitKey(1)

    rett, new_frame = cap.read()

    p0, old_gray, mask = optical_flow(p0, old_gray, new_frame, mask)

    frame = cv2.add(fist_frame, mask)
    cv2.imshow("origin", frame)
    cv2.waitKey(1)

