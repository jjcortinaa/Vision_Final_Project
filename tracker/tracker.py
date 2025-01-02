import numpy as np
import cv2
from picamera2 import Picamera2

dist = 0
focal = 450
pix = 30
width = 4
kernel = np.ones((3,3),'uint8')
font = cv2.FONT_HERSHEY_SIMPLEX
org = (0,20)
fontScale = 0.6
color = (0,0,255)
thickness = 2

picam = Picamera2()
picam.preview_configuration.main.size=(700, 600)
picam.preview_configuration.main.format="RGB888"
picam.preview_configuration.align()
picam.configure("preview")
picam.start()


while True:
    frame = picam.capture_array()

    hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lbound = (10,100,100)
    ubound = (25,255,255)
    mask = cv2.inRange(hsv_img,lbound,ubound)

    d_img = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations = 5)
    cont,hie = cv2.findContours(d_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key=cv2.contourArea,reverse = True)[:1]

    for cnt in cont:
        if cv2.contourArea(cnt)>100 and cv2.contourArea(cnt)<300000:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(np.clip(box,0,[frame.shape[1]-1,frame.shape[0]-1]))
            cv2.drawContours(frame,[box], -1, (255,0,0),3)

            pixels = rect[1][0]
            dist = (width*focal)/pixels

            cv2.putText(frame,str(dist), (110,50),font,fontScale,color,1,cv2.LINE_AA)
    cv2.imshow("DISTANCE",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
