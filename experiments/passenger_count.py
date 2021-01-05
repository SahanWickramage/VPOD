##################################################
##  import dependencies and define parameters   ##
##################################################
import os
from gluoncv import model_zoo, data
import mxnet as mx
import numpy as np
import cv2
from math import sqrt

threshold = 0.4
green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
thickness = 2  # 2 pixel
radius = 3
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
video_path = "test5.mp4"

#########################
##  define functions   ##
#########################
# extract predictions that above the threshold
def predict(frame):
    bboxes_list = []
    scores_list = []
    box_ids_list = []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    xrgb = mx.nd.array(rgb).astype('uint8')
    rgb_nd, xrgb = data.transforms.presets.rcnn.transform_test(xrgb, short=512, max_size=700)
    box_ids, scores, bboxes = model(rgb_nd.as_in_context(mx.gpu(0)))

    bboxes = bboxes[0].asnumpy()
    scores = scores[0].asnumpy()
    labels = box_ids[0].asnumpy()

    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < threshold:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        bboxes_list.append(bbox)
        scores_list.append(scores.flat[i])
        box_ids_list.append(cls_id)

    return (bboxes_list, scores_list, box_ids_list)

# extract predictions of person class
def get_human_box_detection(boxes, scores, classes):
    array_boxes = list()
    for i in range(len(boxes)):
        if int(classes[i]) == 0 and scores[i] > threshold:
            array_boxes.append((int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])))
    return array_boxes


# check near or not using euclidean distance
def is_near(new_point, old_point, check_distance=50):
    real_distance = int(sqrt((new_point[0] - old_point[0])**2 + (new_point[1] - old_point[1])**2))
    if real_distance <= check_distance:
        return True
    else:
        return False

######################
##  main program    ##
######################
# init step
model = model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True, ctx=mx.gpu(0))
fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
output_video_1 = cv2.VideoWriter("video.avi", fourcc1, 20, (640, 480), True)

vs = cv2.VideoCapture(video_path)
count = 0
passengers = []

# main loop
# program keeps a list 'passengers' for entire loop
# list 'passengers' keeps the latest coordinates of centroids of passengers
# number of elements in the list 'passengers' is the number of passengers in the bus
# each iteration program creates list 'centroids' for current frame
# list 'centroids' has the coordinates of centroids of detected passengers in the current frame
# each iteration program updates the latest coordinates of centroids in list 'passengers' with help of 'is_near()' function
# if there is new centroid in the list 'centroids', new centroid will append to list 'passengers'
while True:
    (frame_exists, frame) = vs.read()
    key = cv2.waitKey(1) & 0xFF
    if not frame_exists:
        break
    else:
        if key == ord('q'):
            break
        (boxes, scores, classes) = predict(frame)
        # create list 'centroids'
        centroids = []
        array_boxes_detected = get_human_box_detection(boxes, scores, classes)
        for index, downoid in enumerate(array_boxes_detected):
            xmin, ymin, xmax, ymax = [int(x) for x in array_boxes_detected[index]]
            xc = int((xmin + xmax) / 2)
            yc = int((ymin + ymax) / 2)
            centroids.append((xc, yc))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), green, thickness)
            cv2.circle(frame, (xc, yc), radius, red, -1)

        # no detected passengers in current frame
        if len(centroids) == 0:
            pass
        else:
            # list 'passengers' is empty
            if len(passengers) == 0:
                for i in range(len(centroids)):
                    passengers.append(centroids[i])
            # update the latest coordinates of centroids of passengers
            else:
                _passengers = passengers[:]

                for i in range(len(centroids)):
                    updated = 0
                    for j in range(len(passengers)):
                        if is_near(centroids[i], passengers[j]):
                            _passengers[j] = centroids[i]
                            updated += 1

                    # new centroid found
                    if updated == 0:
                        _passengers.append(centroids[i])

                passengers = _passengers[:]
        if len(passengers) > count:
            count = len(passengers)
        t = "passengers : %u" % count
        cv2.putText(frame, t, (0, 90), font, fontScale, blue, thickness, cv2.LINE_AA)
        cv2.imshow("output", frame)
        output_video_1.write(frame)

# finish
vs.release()
