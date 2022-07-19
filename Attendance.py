import cv2
import numpy
import face_recognition
import os
from datetime import datetime

path='Attendance_Images'
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)

def Encodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodeimg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeimg)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        print(myDataList)
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            timeString=now.strftime('%H:%M:%S')
            dateString=now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{timeString},{dateString}')

for cl in myList:
    current_Img=cv2.imread(f'{path}/{cl}')
    images.append(current_Img)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)



encodeListUnknown=Encodings(images)
print('Encoding Done')

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    ims=cv2.resize(img,(0,0),None,0.25,0.25)
    ims=cv2.cvtColor(ims,cv2.COLOR_BGR2RGB)

    faces_current_frame=face_recognition.face_locations(ims)
    encode_current_frame = face_recognition.face_encodings(ims,faces_current_frame)

    for encodeFace,faceLoc in zip(encode_current_frame,faces_current_frame):
        matches=face_recognition.compare_faces(encodeListUnknown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListUnknown,encodeFace)
        print(faceDis)
        matchIndex=numpy.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1= y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            markAttendance(name)

    cv2.imshow('WebCam',img)
    if cv2.waitKey(10)==13:
        break

cap.release()
cv2.destroyAllWindows()