from classes import *
from OpticalFlow import harris_corners, shi_tomasi

yolo_detect = Detection(0)
video = cv2.VideoCapture(yolo_detect.getSource)

while True:
    ret, frame = video.read()
    boxes, conv_cost = yolo_detect.yoloDetection(frame)

    cv2.imshow("origin", frame)
    cv2.waitKey(1)

    for id, detections in enumerate(boxes):
        shape = detections[2] * detections[3]

        if id is 0: continue
        iou = shape[id]/shape[id-1]
        conv_cost