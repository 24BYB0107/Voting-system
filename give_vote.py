from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time 
from datetime import datetime
from win32.com.client import Dispatch

def speak(str1):
    spaek=Dispatch(("SAPI.SpVoice"))
    speak.S.peak(str1)
    
video= cv2.VideoCapture(1)
facedetect=cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcasacade_frontalface_default.xml')
if not os.path.exists('data/'):
    os.makedirs('data/')
    
with open('data/names.pk1','rb')as f:
    LABELS=pickle.load(f)
    
with open('data/faces_data.pk1','rb') as f:
    FACES=pickle.load(f)
    
knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(FACES,LABELS)
imgBackground=cv2.imread("imgbackground.png")

COL_NAMES=['NAME','VOTE','DATE','TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts=time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("Votes" + ".csv")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        
    cv2.imshow('frame',imgBackground)
    k=cv2.waitKey(1)

