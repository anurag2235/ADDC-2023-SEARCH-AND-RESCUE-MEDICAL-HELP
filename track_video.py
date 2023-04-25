from algorithm.object_detector import YOLOv7
from utils.detections import draw
from tqdm import tqdm
import cv2
yolov7 = YOLOv7()
yolov7.load('coco.weights', classes='coco.yaml', device='cpu') 
video = cv2.VideoCapture(0)
width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('width=',width)
print('height=',height)
fps = int(video.get(cv2.CAP_PROP_FPS))
print('fps=',fps)
frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

if video.isOpened() == False:
	print('[!] error opening the video')

pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)

try:
    while video.isOpened():
        ret, frame = video.read()
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if ret == True:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 
                'aeroKLE', 
                (30, 50), 
                font, 1, 
                (0, 0, 255), 
                2, 
                cv2.LINE_4)
            detections = yolov7.detect(frame, track=True)
            print(detections)
            detected_frame = draw(frame, detections)
            output.write(detected_frame)
            pbar.update(1)
            cv2.imshow('video',detected_frame)
        else:
            break
except KeyboardInterrupt:
    pass

pbar.close()
video.release()
output.release()
yolov7.unload()