import cv2 as cv
import numpy as np

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
#recognizer =  cv.face.LBPHFaceRecognizer_create()
#recognizer.read("trainner.yml")

camera = cv.VideoCapture(0)

while(True):
#Capture frame by Frame   
 _, frame = camera.read()
 gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 faces = face_cascade.detectMultiScale(gray, 1.1, 4)

 for (x, y, w, h) in faces:
     print(x,y,w,h)
     roi_gray = gray[y:y+h, x:x+w]
     roi_color = frame[y:y+h, x:x+w]
     
     #recognize?
     #id_,conf = recognizer.predict(roi_gray)
     #if conf>=45 and conf<=85:
         #print(id_)
     img_item = "my-image.png"
     cv.imshow(img_item, roi_gray)
     
     cv.rectangle(frame, (x,y) , (x+w, y+h) , (255, 0, 0), 3)
 
#Display resulting Frame
 cv.imshow('frame', frame)
 if cv.waitKey(5) == ord('X'):
   break
#When everything done,release camera
camera.release()
cv.destroyAllWindows()