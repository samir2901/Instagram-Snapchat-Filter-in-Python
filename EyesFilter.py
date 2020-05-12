import cv2
import numpy as np
from math import hypot

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

glasses = cv2.imread("FilterImages/glasses.png",-1)

filter_size_ratio = glasses.shape[0] / glasses.shape[1]

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)    
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)    
    
    faces = face_cascade.detectMultiScale(gray,1.5,5)
    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi)
        #print(eyes)   
        for (ex,ey,ew,eh) in eyes:
            roi_eyes = roi[ey:ey + eh, ex:ex + ew]
            glasses2 = cv2.resize(glasses.copy(),(ew, int(ew/filter_size_ratio)))

            for i in range(glasses2.shape[0]):
                for j in range(glasses2.shape[1]):
                    if glasses2[i,j][3] != 0:
                        roi_color[ey+i, ex+j] = glasses2[i,j]        
    

    frame = cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)    
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()