# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 18:26:40 2021

@author: Abhimanyu Das
"""

import cv2
import keras
import os
import time
import threading
from playsound import playsound
import sys
from imutils.video import FileVideoStream, WebcamVideoStream


def loadmodel(modeljson, modelweights, path):
    json_file = open(os.path.join(path, modeljson), 'r')
    model = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(model)
    model.load_weights(os.path.join(path, modelweights))
    return model



def alert():
    for i in range(2):
        playsound('beep-07.wav')
        time.sleep(1)

def predictSample(img):
    if(img.size):
        img = cv2.resize(img, (224,224))
        img = img.reshape(1,224,224,3)
        return model.predict(img)
        #plt.imshow(origin)
        
        
def detect(frame):
    global stream_list, t1
    time.sleep(0.2)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    
    if (len(faces) != 0):
        (x,y,w,h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        try:
            eyes = eye_cascade.detectMultiScale(roi_gray)
        except:
            return
        #cv2.putText(frame,'Face',(x+w,y+h),font,1,(250,250,250),2,cv2.LINE_AA)   #display a text label 'Face'
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)       #draw rectangle around face
        if(len(eyes) == 2):
            
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
                roi = roi_color[ey:ey+eh, ex:ex+ew]
                #cv2.imshow('',roi)
                op = predictSample(roi)
                op = 1 if op > 0.5 else 0
                
                if(len(stream_list) < 10):
                    stream_list.append(op)
                else:
                    total_ones = sum(stream_list) #no of frames where the eyes were open (detected)
                    stream_list.append(op)
                    stream_list = stream_list[1:]
                    #print(total_ones)
                    if(total_ones<5):
                        putdrowsy()
                    if(total_ones < 5 and not t1.is_alive()):
                        t1 = threading.Thread(target=alert)
                        t1.start()
                        
def putdrowsy():
    global frame
    cv2.putText(frame,'Drowsy!!!',(100,100),font,1,(0,0,255),2,cv2.LINE_AA)
                        

def makeboxes():
    global frame
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    if (len(faces) != 0):
        (x,y,w,h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        try:
            eyes = eye_cascade.detectMultiScale(roi_gray)
        except:
            return
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        if(len(eyes) == 2):
            
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
        
        
    
    
                
                

    

if __name__ == '__main__':
    cap = None
    if(len(sys.argv) == 2):
        arg = sys.argv[1]
        #print(sys.argv)
        fvs = FileVideoStream(arg, queue_size=50).start()
    else:
        cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture(0)
    
    time.sleep(1)
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_COMPLEX
    model = loadmodel('model.json', 'model.h5', './')
    buffer = 0
    t1 = threading.Thread(target=alert)
    detect_thread = threading.Thread(target=detect)
    makeboxes_thread = threading.Thread(target=makeboxes)
    stream_list = []
    
    
    
    while (cap or fvs.more()):
        #time.sleep(0.01)
        if(cap):
            _, frame = cap.read()
        else:
            frame = fvs.read()
            #time.sleep(0.05)
            
    
            
        
            
        
        
        if(not detect_thread.is_alive()):
            detect_thread = threading.Thread(target=detect, args=(frame,))
            detect_thread.start()
            
        # if(not makeboxes_thread.is_alive()):
        #     makeboxes_thread = threading.Thread(target=makeboxes)
        #     makeboxes_thread.start()
        
        makeboxes()
        
            
        cv2.imshow('Frame', frame)
     
       
        
        key = cv2.waitKey(1)
        if (key == 27):
            if(cap):
                cap.release()
            else:
                fvs.stop()
            #cap.release()
            cv2.destroyAllWindows()
            break
        
        
