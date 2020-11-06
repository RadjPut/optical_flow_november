from classes import *
from OpticalFlow import harris_corners, shi_tomasi

sort = SORT()
yolo_detect = Detection(0)
video = cv2.VideoCapture(yolo_detect.getSource)
# fist_iter = False
p0 = np.array([[[0, 0]]])
# a = p0[0][0][0]
#iou_boxes = []
iou_threshold = 0.3
id = 0

untracked_detections = []
tracked_detections = []

while True:
    ret, frame = video.read()
    boxes, confidence = yolo_detect.yoloDetection(frame)
    for p, detection in enumerate(boxes):

      #  k += 1  # calculate boxes
        #if k % 3 == 0:
        untracked_detections.append([detection, id])
        if len(untracked_detections) > 1:
            for i in range(len(untracked_detections)-1):
                iou = sort.iou_bb(untracked_detections[i][0], untracked_detections[i+1][0])
                if iou > iou_threshold:
                    tracked_detections.append(untracked_detections[-1])
                else:
                    id += 1
                    tracked_detections.append(untracked_detections[-1])

                print iou

    cv2.imshow("origin", frame)
    cv2.waitKey(1)
    # if len(boxes) != 0:
    # some = boxes[0]
    # frame = frame[some[1]:(some[1]+some[3]), some[0]:(some[0]+some[2])]

    if len(boxes) == 0:
        #cv2.destroyAllWindows()
        #video.release()
         #a = boxes[0][0]
        continue

    if (boxes[0][0] > p0[0][0][0] < boxes[0][1]) | (boxes[0][0]+boxes[0][2] < p0[0][0][0] > boxes[0][1]+boxes[0][3]):
        for detections in boxes:
            # if fist_iter:
            #ret, old_frame = video.read()

            p0, old_gray, mask = yolo_detect.shiTomasi(frame, detections)  # check p0 NaN values\111 Error nontype object iterable
            # cv2.imshow("asdasd", old_gray)

        # some = None
        # cv2.imshow("next frame", old_gray)
        # cv2.imshow("croped frame", new_old_gray)
        # cv2.waitKey(1)
    else:
        ret, new_frame = video.read()
        old_gray, p0, good_new, good_old = yolo_detect.opticalFlow(new_frame, p0, old_gray)
        yolo_detect.draw_optical_flow(mask, frame, good_new, good_old, boxes, tracked_detections)

        untracked_detections.remove(untracked_detections[0])
        tracked_detections.remove(tracked_detections[0])

        tracked_detections = []
        #untracked_detections = []
        frame = cv2.add(frame, mask)
        cv2.imshow("origin", frame)
        cv2.waitKey(1)
