import cv2

cap = cv2.VideoCapture(0)
ret,frame = cap.read()
i=1

while ret:
	ret,frame = cap.read()
	cv2.imshow("frame",frame)
	k = cv2.waitKey(10)
	if k == 32:
		cv2.imwrite("hamza"+str(i)+".jpg",frame)
		i+=1
	elif k == 27: 
		break
	
	
