import cv2
import time
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to read camera feed")


import os
output_dir = 'output_image'
os.makedirs(output_dir, exist_ok=True)


import supervision as sv
from ultralytics import YOLOv10

model = YOLOv10(f'best.pt') # yolov10s.pt
#image = cv2.imread(f'source/')
#results = model(image)[0]

box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
font = cv2.FONT_HERSHEY_PLAIN
# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break;
    
    new_frame_time = time.time() 
    # Calculating the fps 
  
    # fps will be number of frame processed in given time frame 
    # since their will be most of time error of 0.001 second 
    # we will be subtracting it to get more accurate result 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
  
    # converting the fps into integer 
    fps = int(fps) 
  
    # converting the fps to string so that we can display it on frame 
    # by using putText function 
    fps = str(fps) 
  
    # putting the FPS count on the frame 
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    annotated_image = box_annotator.annotate(scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    #sv.plot_image(annotated_image)
    cv2.imshow('Webcam', frame)
    k = cv2.waitKey(1)
    
# When everything done, release 
# the video capture object 
cap.release() 
  
# Closes all the frames 
cv2.destroyAllWindows() 