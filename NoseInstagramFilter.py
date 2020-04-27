import cv2
import numpy as np
from mtcnn import MTCNN
from math import hypot


cap = cv2.VideoCapture(0)
nose = cv2.imread("FilterImages/nosefilter3.png",-1)
detector = MTCNN()
filter_size_ratio = nose.shape[0]/nose.shape[1]

while True:
    _, frame = cap.read()
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) 
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)

    result = detector.detect_faces(img)
    
    if len(result) != 0:              
        point_nose = result[0]['keypoints']['nose'] #center point of nose
        points_mouth = [result[0]['keypoints']['mouth_left'], result[0]['keypoints']['mouth_right']]
        #print(points_mouth[0], points_mouth[1])

        nose_width = int(hypot(points_mouth[0][0] - points_mouth[1][0], points_mouth[0][1] - points_mouth[1][1]) * 1.2)
        nose_height = int(nose_width * filter_size_ratio)    
        
        nose2 = cv2.resize(nose.copy(),(nose_width,nose_height))        
        top_left = (int(point_nose[0] - nose_width/2), int(point_nose[1] - nose_height/2))        
        #roi = frame[top_left[1] : top_left[1] + nose_height, top_left[0] : top_left[0] + nose_width]

        for i in range(nose_height):
            for j in range(nose_width):
                if nose2[i,j][3] != 0: #checking if pixel value is transparent
                    frame[top_left[1] + i, top_left[0] + j] = nose2[i,j]       
        
        
        
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
