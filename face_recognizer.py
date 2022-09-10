import os

import cv2
import numpy as np
from keras_facenet import FaceNet


def who_is_it(embeeds, database):
    min_dist = 100
    identity = "<UKN>"
    for (name, db_enc) in database.items():
        for enc in db_enc:
            dist = np.linalg.norm(embeeds - enc)
            if dist < min_dist:
                min_dist = dist
                identity = name

    if min_dist > 0.6:
        identity = "<UKN>"
    return identity


def load_dataset(model):
    database = {}
    for person in os.listdir('dataset'):
        database[person] = []
        for pic in os.listdir('dataset' + '/' + person):
            img = cv2.imread('dataset' + '/' + person + '/' + pic)
            database[person].append(model.extract(
                img, threshold=0.95)[0]['embedding'])

    return database


# Constante
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
bottomLeftCornerOfText = (55, 25)


# Load enconding face model , FaceNet
model = FaceNet()

# Creating the database of encoded faces
database = load_dataset(model)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

while ret:
    detections = model.extract(frame, threshold=0.95)
    for detection in detections:
        x, y, h, w = detection['box']
        identity = who_is_it(detection['embedding'], database)
        cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 0, 255), 2)
        cv2.putText(frame, identity, (x-10, y-10), font,
                    fontScale, fontColor, lineType)

    cv2.imshow('img', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
    ret, frame = cap.read()
cv2.destroyAllWindows()
