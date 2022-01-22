import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

currentid = 0
labelsid = {}
labely = []
trainx = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if not label in labelsid:
                labelsid[label] = currentid
                currentid += 1
            id_ = labelsid[label]
            pil_image = Image.open(path).convert("L") # grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")

            faces = face_cascade.detectMultiScale(image_array, 1.5, 5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                trainx.append(roi)
                labely.append(id_)


print(labely)
print(trainx)

with open("./labels.pickle", 'wb') as f:
    pickle.dump(labelsid, f)

recognizer.train(trainx, np.array(labely))
recognizer.save("recognizers/face-trainner.yml")