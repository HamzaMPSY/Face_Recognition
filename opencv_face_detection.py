import cv2
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json

def img_to_encoding(img, model):
    img = img[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def loadModel():
	json_file = open('face_encoding.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("face_encoding.h5")
	print("Loaded model from disk")
	return loaded_model

def who_is_it(image, database, model):
	encoding = img_to_encoding(image, model)
	min_dist = 100
	identity = "<UKN>"
	for (name, db_enc) in database.items():
		for enc in db_enc:
			dist = np.linalg.norm(encoding - enc)
			if dist < min_dist:
				min_dist = dist
				identity = name


	if min_dist > 0.6:
		print("Not in the database.")
		identity = "<UKN>"
	else:
		print ("it's " + str(identity) + ", the distance is " + str(min_dist))

	return identity

# Constante
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
bottomLeftCornerOfText = (55,25)

# Load Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load enconding face model , FaceNet
model = loadModel()

# Creating the database of encoded faces
database = {}
database['hamza'] = []
for img in os.listdir('dataset'):
	img = cv2.imread('dataset/'+img,1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	x,y,w,h = faces[0]
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
	img = img[y:y+h, x:x+w]
	img = cv2.resize(img,(96,96))
	database['hamza'].append(img_to_encoding(img, model))

cap = cv2.VideoCapture(0)
ret,frame = cap.read()


while ret:	
	
	# Convert image to graysclae
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
	    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
	    img = frame[y:y+h, x:x+w]
	    img = cv2.resize(img,(96,96))
	    identity = who_is_it(img,database,model)
	    cv2.putText(frame,identity,(x-10,y-10), font, fontScale,fontColor,lineType)

	#cv2.imwrite("detection.jpg",img)
	cv2.imshow('img',frame)
	key = cv2.waitKey(10)
	if key == 27:   
		break
	ret,frame = cap.read()
cv2.destroyAllWindows()
