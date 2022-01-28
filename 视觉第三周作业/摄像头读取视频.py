import numpy as np
import cv2

cap=cv2.VideoCapture(0)
while(True):#Captureframe-by-frame15
    ret,frame=cap.read()
    #Ouroperationsontheframecomehere
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Displaytheresultingframe
    cv2.imshow('frame',gray)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
        #Wheneverythingdone,releasethecapture
        cap.release()
        cv2.destroyAllWindows()