import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to read camera feed")


import os
output_dir = 'output_image'
os.makedirs(output_dir, exist_ok=True)


import supervision as sv
from ultralytics import YOLOv10

model = YOLOv10(f'best.pt')
#image = cv2.imread(f'source/')
#results = model(image)[0]

box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()


while True:
    ret, frame = cap.read()
    if not ret:
        break;
    
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