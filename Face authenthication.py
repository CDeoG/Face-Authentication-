import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("./labels.pickle", 'rb') as f:
  org_labels = pickle.load(f)
  labels = {v:k for k,v in org_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x, y, w, h) in faces:
      roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
      roi_color = frame[y:y+h, x:x+w]

      # recognize? deep learned model predict keras tensorflow pytorch scikit learn
      idd, conf = recognizer.predict(roi_gray)
      if conf>=4 and conf <= 85:
        name = labels[idd]
        cv2.putText(frame, name, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

      img_item = "7.png"
      cv2.imwrite(img_item, roi_color)
      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Camera',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
