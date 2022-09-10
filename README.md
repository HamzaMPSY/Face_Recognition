# Face Recognition
A python script that can detect and recognize faces in videos, to detect faces i use keras_facenet library that gets a detection dict for each face in an image. Each one has the bounding box and face landmarks (from mtcnn.MTCNN) along with the embedding from FaceNet. in every frame i detect the face and i encode it than i compare it with the images in database

To test it with your own face add a folder in in dataset folder with your name and add some pics of your face in it 

you can use take_pics to take some pics (press space to take a pic)
![img](/images/image1.PNG)

## Requirements
```
openCV
numpy
tensorflow
keras
keras_facenet
```
Use the pip commande to install this requirements : pip install "package name"
