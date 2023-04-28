from algorithm.object_detector import YOLOv7
from utils.detections import draw
import json
import cv2
import imutils
import numpy as np
from mss import mss
from PIL import Image
sct = mss()
yolov7 = YOLOv7()
yolov7.load('coco.weights', classes='coco.yaml', device='cpu') # use 'gpu' for CUDA GPU inference
try:
    while True:
        w, h = 640, 640
        monitor = {'top': 0, 'left': 0, 'width': w, 'height': h}
        img = Image.frombytes('RGB', (w,h), sct.grab(monitor).rgb)
        img_np = np.array(img)
        frame = img_np
        count=0
        A=True
        if A == True:      
            detections = yolov7.detect(frame)
            detected_frame = draw(frame, detections)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(detected_frame, 
                'aeroKLE', 
                (30, 50), 
                font, 1, 
                (0, 0, 255), 
                2, 
                cv2.LINE_4)
            cv2.putText(detected_frame, 
                'SAE ADDC 2023', 
                (30, 100), 
                font, 1, 
                (0, 0, 255), 
                2, 
                cv2.LINE_4)
            cv2.putText(detected_frame, 
                'NUMBER OF PEOPLE IN NEED', 
                (150, 50), 
                font, 1, 
                (0, 255, 0), 
                2, 
                cv2.LINE_4)          
            if len(detections)!=0:
                i=0
                for items in detections:
                   identity=detections[i]['class']
                   if identity=='person':
                    count=count+1
                   i=i+1
                print("DETECTED PERSON = ",count)
            cv2.putText(detected_frame, 
                str(count), 
                (500, 100), 
                font, 1, 
                (0, 255, 0), 
                2, 
                cv2.LINE_4)     
            cv2.imshow('webcam', detected_frame)
            cv2.waitKey(1)
        else:
            break
except KeyboardInterrupt:
    pass
webcam.release()
print('[+] webcam closed')
yolov7.unload()
