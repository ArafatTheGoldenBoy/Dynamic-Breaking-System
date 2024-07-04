import cv2

cap = cv2.VideoCapture("video3.mp4")

import supervision as sv
from ultralytics import YOLOv10
from ultralytics.utils.plotting import Annotator
import math
import numpy as np
import matplotlib.pyplot as plt
model = YOLOv10(f'yolov10b.pt')
# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
# Center point and pixel per meter for distance calculation
center_point = (0, h)
pixel_per_meter = 10
data = {}
labels = []
class_counts_over_time = {}
speed_over_time = {}
distance_over_time = {}
frame_count = 0
# Colors for text and bounding box
txt_color, txt_background, bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))


while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break;
    frame_count += 1
    results = model.track(frame, persist=True)
    # Object detection for distance estimation
    annotator = Annotator(frame, line_width=2)
    annotated_frame = results[0].plot()
    lowest_distances = []
    lowest_ditected_object = []
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            cls_name = model.names[int(cls)]
            annotator.box_label(box, label=str(track_id), color=bbox_clr)
            annotator.visioneye(box, center_point)

            x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)  # Bounding box centroid

            distance = (math.sqrt((x1 - center_point[0]) ** 2 + (y1 - center_point[1]) ** 2)) / pixel_per_meter
            lowest_ditected_object.append(cls_name)
            lowest_distances.append(distance)
            text_size, _ = cv2.getTextSize(f"Distance: {distance:.2f} m", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), txt_background, -1)
            cv2.putText(annotated_frame, f"Distance: {distance:.2f} m", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, txt_color, 2)
    # Clear counts for next frame
    if len(lowest_distances) != 0:
        lowest_distance = min(lowest_distances)
        lowest_distance_position = lowest_distances.index(lowest_distance)
        print("+++++++++++++++++++++++++++++++++ lowest_distance +++++++++++++++++++++++++++++++++", lowest_distance)
        print("+++++++++++++++++++++++++++++++++ lowest_distance_position +++++++++++++++++++++++++++++++++", lowest_distance_position)
        print("+++++++++++++++++++++++++++++++++ Ditected Object +++++++++++++++++++++++++++++++++", lowest_ditected_object[lowest_distance_position])
        #print("+++++++++++++++++++++++++++++++++ Ditected Object name +++++++++++++++++++++++++++++++++",)
        #for x in lowest_distances:
            #print(x)
    data = {}
    lowest_distances = []
    lowest_ditected_object = []
    cv2.imshow('Webcam', annotated_frame)
    k = cv2.waitKey(1)
# When everything done, release 
# the video capture object 
cap.release() 
  
# Closes all the frames 
cv2.destroyAllWindows() 