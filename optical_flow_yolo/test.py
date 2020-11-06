import numpy as np

k = 1
if k >= 2 & k % 5 == 0:
    print 0
else:
    print k

detections = [[1, 2, 3], [1, 2, 3]]
trackers = [[1, 2, 3]]

iou_mat = np.zeros((len(detections), len(trackers)), dtype=float)
iou = 0
def iou(trk, det):
    return k
for t, trk in enumerate(trackers):
    for d, dtc in enumerate(detections):
        k+=1
        iou_mat[t, d] = iou(trk, dtc)
