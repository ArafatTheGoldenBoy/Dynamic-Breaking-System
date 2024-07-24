import time
import cv2

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
filename = 'labels.txt'
with open(filename, 'rt') as spt:
    classLabels = spt.read().rstrip('\n').split('\n')
    
    
model.setInputSize(320, 320)  #greater this value better the reults but slower. Tune it for best results
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

    
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN
# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0
while(True):

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
    classIndex, confidence, bbox = model.detect(frame , confThreshold=0.65)  #tune the confidence  as required
    if(len(classIndex) != 0):
        for classInd, boxes in zip(classIndex.flatten(), bbox):
            cv2.rectangle(frame, boxes, (255, 0, 0), 2)
            if(1 <= classInd <=80 ):
                cv2.putText(frame, classLabels[classInd-1], (boxes[0] + 10, boxes[1] + 40), font, fontScale = 1, color=(0, 255, 0), thickness=2)

    cv2.imshow('result', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

