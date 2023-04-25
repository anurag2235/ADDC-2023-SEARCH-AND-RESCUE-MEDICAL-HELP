from algorithm.object_detector import YOLOv7
from utils.detections import draw
import json
import cv2
import imutils
yolov7 = YOLOv7()
yolov7.load('coco.weights', classes='coco.yaml', device='cpu') # use 'gpu' for CUDA GPU inference
webcam = cv2.VideoCapture(0)        #give rtsp link here inplace of zeroes
if webcam.isOpened() == False:
	print('[!] error opening the webcam')
try:
    while webcam.isOpened():
        ret, frame = webcam.read()
        count=0
        frame =imutils.resize(frame, width=900)
        frame =imutils.resize(frame, height=900)
        if ret == True:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 
                'aeroKLE', 
                (30, 50), 
                font, 1, 
                (0, 0, 255), 
                2, 
                cv2.LINE_4)
            cv2.putText(frame, 
                'SAE ADDC 2023', 
                (30, 100), 
                font, 1, 
                (0, 0, 255), 
                2, 
                cv2.LINE_4)    
            cv2.putText(frame, 
                'NUMBER OF PEOPLE IN NEED', 
                (700, 50), 
                font, 1, 
                (0, 255, 0), 
                2, 
                cv2.LINE_4)         
            detections = yolov7.detect(frame)
            detected_frame = draw(frame, detections)
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
                (1080, 100), 
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