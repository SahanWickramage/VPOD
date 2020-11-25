# import dependencies
import os
from gluoncv import model_zoo, data
import mxnet as mx
import numpy as np
import cv2
from math import sqrt

# define parameters
threshold = 0.4
green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
thickness = 2  # 2 pixel
radius = 3
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
video_path = "test.mp4"

# define functions
def predict(frame):
    bboxes_list = []
    scores_list = []
    box_ids_list = []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    xrgb = mx.nd.array(rgb).astype('uint8')
    # x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)
    rgb_nd, xrgb = data.transforms.presets.rcnn.transform_test(xrgb, short=512, max_size=700)
    box_ids, scores, bboxes = model(rgb_nd.as_in_context(mx.gpu(0)))

    # img = gcv.utils.viz.cv_plot_bbox(frame, bboxes[0], scores[0], box_ids[0], class_names=net.classes)
    # gcv.utils.viz.cv_plot_image(img)

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

def get_human_box_detection(boxes, scores, classes):
    array_boxes = list()
    for i in range(len(boxes)):
        if int(classes[i]) == 0 and scores[i] > threshold:
            array_boxes.append((int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])))
    return array_boxes

# select the model
model = model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True, ctx=mx.gpu(0))
fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
output_video_1 = cv2.VideoWriter("video.avi", fourcc1, 20, (640, 480), True)

def is_near(new_point, old_point, check_distance=50):
    real_distance = int(sqrt((new_point[0] - old_point[0])**2 + (new_point[1] - old_point[1])**2))
    if real_distance <= check_distance:
        return True
    else:
        return False

# init
vs = cv2.VideoCapture(video_path)
count = 0
passengers = []
# main loop
while True:
    (frame_exists, frame) = vs.read()
    key = cv2.waitKey(1) & 0xFF
    if not frame_exists:
        break
    else:
        if key == ord('q'):
            break
        (boxes, scores, classes) = predict(frame)
        centroids = []
        line = ""
        array_boxes_detected = get_human_box_detection(boxes, scores, classes)
        for index, downoid in enumerate(array_boxes_detected):
            xmin, ymin, xmax, ymax = [int(x) for x in array_boxes_detected[index]]
            xc = int((xmin + xmax) / 2)
            yc = int((ymin + ymax) / 2)
            centroids.append((xc, yc))
            if line == "":
                line = "%u,%u" % (xc, yc)
            else:
                line = "%s %u,%u" % (line, xc, yc)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), green, thickness)
            cv2.circle(frame, (xc, yc), radius, red, -1)

        if len(centroids) == 0:
            pass
        else:
            if len(passengers) == 0:
                for i in range(len(centroids)):
                    passengers.append(centroids[i])
            else:
                _passengers = passengers[:]

                for i in range(len(centroids)):
                    updated = 0
                    for j in range(len(passengers)):
                        if is_near(centroids[i], passengers[j]):
                            _passengers[j] = centroids[i]
                            updated += 1

                    if updated == 0:
                        _passengers.append(centroids[i])

                passengers = _passengers[:]
        if len(passengers) > count:
            count = len(passengers)
            print(count)
        t = "passengers : %u" % count
        cv2.putText(frame, t, (0, 90), font, fontScale, blue, thickness, cv2.LINE_AA)
        cv2.imshow("output", frame)
        output_video_1.write(frame)

# finish
vs.release()
