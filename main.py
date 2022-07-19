import cv2
import numpy as np
import face_recognition
imgSk=face_recognition.load_image_file('ImagesBasic/sk.jpg')
imgSk=cv2.cvtColor(imgSk,cv2.COLOR_BGR2RGB)
imgSk_test=face_recognition.load_image_file('ImagesBasic/sk_test.jpg')
imgSk_test=cv2.cvtColor(imgSk_test,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgSk)[0]
encodeSk=face_recognition.face_encodings(imgSk)[0]
cv2.rectangle(imgSk,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)

faceLoc_test=face_recognition.face_locations(imgSk_test)[0]
encodeSk_test=face_recognition.face_encodings(imgSk_test)[0]
cv2.rectangle(imgSk_test,(faceLoc_test[3],faceLoc_test[0]),(faceLoc_test[1],faceLoc_test[2]),(0,255,0),2)

answer=face_recognition.compare_faces([encodeSk],encodeSk_test)
dis_face=face_recognition.face_distance([encodeSk],encodeSk_test)
print(answer,dis_face)
cv2.putText(imgSk_test,f'{answer} {round(dis_face[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)

cv2.imshow('sk',imgSk)
cv2.imshow('sk_test',imgSk_test)

cv2.waitKey(0)

