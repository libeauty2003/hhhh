import cv2
import numpy as np

img=np.zeros((512,512,3),np.uint8)
cv2.circle(img,(100,250),100,(234,23,255),0)
cv2.rectangle(img,(44,60),(400,100),(135,255,0),-1)
cv2.rectangle(img,(0,200),(200,500),(255,244,222),1)
while(1):
    cv2.imshow("image",img)
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break
cv2.destroyAllWindows()

