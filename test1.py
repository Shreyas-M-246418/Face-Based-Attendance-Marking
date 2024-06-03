import time
from sklearn.neighbors import KNeighborsClassifier as knnn
import cv2
import pickle
import numpy as np
import os
import csv
from datetime import datetime

vid=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('C:/Users/shrey/Desktop/Data/face_reco/data/haarcascade_frontalface_default.xml')

with open('data/names.pkl','rb') as f:
    labels=pickle.load(f)

with open('data/faces_data.pkl','rb') as f:
    faces=pickle.load(f)

knn=knnn(n_neighbors=5)
knn.fit(faces, labels)

col_names=["Names","TIME"]
w=0
pred=""
global output


while(True):
    ret,frame=vid.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h,x:x+w,:]
        res_immg=cv2.resize(crop_img,(50,50)).flatten().reshape(1,-1)
        output=knn.predict(res_immg)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timeSt=datetime.fromtimestamp(ts).strftime("%H-%M-%S")
        exist=os.path.isfile("Attendance/Attendance_"+date+".csv")
        cv2.putText(frame,str(output[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        att=[str(output[0]),str(timeSt)]
    cv2.imshow("Frame", frame)
    k=cv2.waitKey(1)
    if str(output[0])!=pred or w==0 or pred =="":
        if exist:
            with open("Attendance/Attendance_"+date+".csv","+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(att)
                w=1
            csvfile.close()
        else:
            with open("Attendance/Attendance_"+date+".csv","+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(col_names)
                writer.writerow(att)
                w=1
            csvfile.close()
    if k==ord('q'): 
        break
    pred=str(output[0])
vid.release()
cv2.destroyAllWindows()
