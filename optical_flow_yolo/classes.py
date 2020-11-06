import cv2
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
#from yolov4.tf import YOLOv4


class Detection(object):
    #  weights_file=".\yolov4-tiny_last1609.weights", config_file=".\yolov4-tiny1609.cfg"
    #  weights_file=".\yolov3-tiny_last_now.weights", config_file=".\yolov3-tiny_last_cfg.cfg"
    # weights_file=".\yolov4-tiny1609_best2.weights", config_file="yolov4-tiny16092.cfg"
    def __init__(self, source_file=0, names_object=r".\obj.names", weights_file=r".\yolov3-tiny_last_now.weights", config_file=r".\yolov3-tiny_last_cfg.cfg"):
        self.__weights = weights_file
        self.__config = config_file
        self.__source = source_file
        self.__names = names_object
        self.__frame = None
        self.__label = None
        self.__result_tuple = []
        # optical_flow params
        self.__feature_params = dict(maxCorners=100,
                                     qualityLevel=0.3,
                                     minDistance=7,
                                     blockSize=7)
        self.__lk_params = dict(winSize=(15, 15),
                                maxLevel=2,
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        with open(self.__names, "r") as f:
            self.__classes = [line.strip() for line in f.readlines()]

        self.__colors = np.random.uniform(0, 255, size=(len(self.__classes), 3))

        # load YOLO pretrained weights and config file
        self.__net = cv2.dnn.readNet(self.__weights, self.__config)
        # self.__net = cv2.dnn_DetectionModel(self.__weights, self.__config)
        # self.__net.setInputSize(416, 416)
        # self.__net.setInputScale(1.0/ 255)

        if self.__source == 0:  # for webcam
            self.__cap = 0
        elif self.__source == 1:  # for rtsp stream
            self.__cap = "rtsp://cam:jhg23dfc@178.150.141.135:1555/Streaming/Channels/101"
        #  print(2)
        elif self.__source == 2:  # for img
            self.__cap = r"C:\Users\r.pedan\ComputerVison\car.jpg"
        else:
            raise Exception  # nado dopisat'

    def yoloDetection(self, frame):
        class_ids = []
        confidences = []
        boxes = []


        if self.__source == 2:
           frame = cv2.imread(self.__cap)
    #    else:
    #       flag, frame = self.__cap.read()

        layer_names = self.__net.getLayerNames()
        self.__outputlayers = [layer_names[i[0] - 1] for i in self.__net.getUnconnectedOutLayers()]

        height, width, channels = frame.shape
    # detecting objects 0.00392
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.__net.setInput(blob)
        outs = self.__net.forward(self.__outputlayers)

    # Showing info on screen/ get confidence score of algorithm in detecting an object in blob

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)   # width, height
                    h = int(detection[3] * height)

                    # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                    # rectangle co-ordinaters
                    x = int(center_x - w / 2)   # coordinates of upper left angle
                    y = int(center_y - h / 2)
                    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                    boxes.append([x, y, w, h])  # put all rectangle areas
                    confidences.append(float(confidence))  # how confidence was that object detected and show that percentage
                    class_ids.append(class_id)  # name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

#        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
#                self.__label = str(self.__classes[class_ids[0]])
#                color = self.__colors[0]
#                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#                cv2.putText(frame, self.__label, (x, y + 30), font, 1, (255, 255, 255), 2)
#                self.__frame = frame
#                self.__result_tuple.append(self.__label)

        # cv2.imshow("Frame", frame)
        # cv2.waitKey(1)
        return boxes, confidence

    def shiTomasi(self, old_frame, detections):
        # old_frame = old_frame.copy()[some[1]:(some[1] + some[3]), some[0]:(some[0] + some[2])]

        # ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        #    a = detections[0]
        new_old_gray = old_gray.copy()[detections[1]:(detections[1] + detections[3]), detections[0]:(detections[0] + detections[2])]

        p0 = cv2.goodFeaturesToTrack(new_old_gray, mask=None, **self.__feature_params)

        if p0 is None:
            return

        for elem in p0:
            # a = 1
            elem[0][0] += detections[0]
            elem[0][1] += detections[1]

        mask = np.zeros_like(old_frame)
        # corners_img = np.int0(p0)
        # for corners in corners_img:
        #     x, y = corners.ravel()
        #     # Circling the corners in green
        #     cv2.circle(old_gray, (x, y), 3, [0, 255, 0], -1)
        #     # count_tomasi += 1

        # Create a mask image for drawing purposes
       # mask = np.zeros_like(old_frame, mask)
        return p0, old_gray, mask

    def opticalFlow(self, frame, p0, old_gray):

        # retur, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.__lk_params)

        if st is None:
            return None
        #      p0, old_gray, mask = Detection.shiTomasi(self,old_frameframe)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        #for i, (new, old) in enumerate(zip(good_new, good_old)):
        #    a, b = new.ravel()
        #    c, d = old.ravel()
        #    mask = cv2.line(mask, (a, b), (c, d), color, 2)
        #    frame = cv2.circle(frame, (a, b), 5, color, -1)
        #    frame = cv2.circle(frame, (c, d), 5, color1, -1)
        #img = cv2.add(frame, mask)
        #
        #cv2.imshow('frame', img)
        #cv2.waitKey(1)
      #  k = cv2.waitKey(30) & 0xff
      #  if k == 27:
            #break

        # Now update the previous frame and previous points
       # p2 = p0
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        return old_gray, p0, good_new, good_old

      #  cv2.destroyAllWindows()
       # cap.release()

    def draw_optical_flow(self, mask, frame, good_new, good_old, boxes, tracked_detections):

        # Create some random colors
        color = (0, 255, 0)
        color1 = (255, 255, 0)
        color2 = (255, 0, 0)  # untracked detections

        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            self.__label = str("Car")
            color = self.__colors[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, self.__label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(tracked_detections[0][1]), (x, y + 60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            self.__frame = frame
            self.__result_tuple.append(self.__label)

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color, 2)
            frame = cv2.circle(frame, (a, b), 5, color, -1)
            frame = cv2.circle(frame, (c, d), 5, color1, -1)

    @property
    def getObjectName(self):
        return self.__label

    @property
    def getFrame(self):
        return self.__frame

    @property
    def getSource(self):
        return self.__cap

    @property
    def getResultTuple(self):
        return self.__result_tuple


class SORT(object):

    def __init__(self):
        # Initialize parametes for tracker (history)
        self.id = 0  # tracker's id
        self.box = []  # list to store the coordinates for a bounding box
        self.hits = 0  # number of detection matches
        self.no_losses = 0  # number of unmatched tracks (track loss)

        # Initialize parameters for Kalman Filtering

        self.boxes = []
        self.dt = 1  # time interval

    def iou_bb(self, boxA, boxB):

        # boxA = [upper_left_point, bottom_right_point, w, h]

        xB = max(boxA[0], boxB[0])
        yB = max(boxA[1], boxB[1])
        xD = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yD = min(boxA[1] + boxB[3], boxB[1] + boxB[3])
        # compute the area of intersection rectangle
        inter_area = max(0, xD - xB) * max(0, yD - yB)

        union_area = (boxA[2] * boxA[3]) + (boxB[2] * boxB[3]) - inter_area

        iou = inter_area / float(union_area)

        return iou

    def shape_bb(self, boxA, boxB):

        boxA_shape = boxA[2] * boxA[3]
        boxB_shape = boxB[2] * boxB[3]

        shape_score = boxA_shape / float(boxA_shape)

        return shape_score

    def assign_detections_to_trackers(self, trackers, detections, iou_threshold=0.6):

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for t, trk in enumerate(trackers):
            for d, dtc in enumerate(detections):
                iou_matrix[t, d] = self.iou_bb(trk, dtc)

        matched_indices = linear_assignment(-iou_matrix)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    #def ()