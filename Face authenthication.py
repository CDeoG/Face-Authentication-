import cv2 as cv
import numpy as np


face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer =  cv.face.LBPHFaceRecognizer_create();
recognizer.read("recognizers/face-trainner.yml")



font = cv.FONT_HERSHEY_DUPLEX

camera = cv.VideoCapture(0)

while(True):
#Capture frame by Frame   
 ret, im =camera.read()
 gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
 faces = face_cascade.detectMultiScale(gray, 1.3, 5)

 for (x, y, w, h) in faces:
     print(x,y,w,h)
     roi_gray = gray[y:y+h, x:x+w]
     roi_color = im[y:y+h, x:x+w]
     
     #recognize?
     Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
     if(conf<55):
        if(Id==1):
          Id="Deo"
        if(Id==2):
          Id="name"
        if(Id==4):
          Id="name"
         
     else:
          Id="Unknown"  

     cv.putText(im,str(Id),(x,y+h),font,3,255)
        
     img_item = "my-image.png"
     cv.imshow(img_item, roi_gray)
     
     cv.rectangle(im, (x,y) , (x+w, y+h) , (255, 0, 0), 3)
 
#Display resulting Frame
 cv.imshow('im', im)
 if cv.waitKey(10) & 0xFF==ord('q'):
   break
#When everything done,release camera
camera.release()
cv.destroyAllWindows()