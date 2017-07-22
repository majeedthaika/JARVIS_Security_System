import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
import imutils
import dlib
from imutils import face_utils
import argparse
import time
import normalizer
import os
#from __future__ import print_function


def onLayMessage(string,location,frame,color="white"):
    if color == "white":
        colorcode = (255,255,255)
    elif color == "red":
        colorcode = (0,0,255)
    elif color == "blue":
        colorcode = (255,0,0)
    else:
        print "onLayMessage:defaulting to white"
        colorcode = (255,255,255)
    if location == "bottom":
        cv2.putText(frame, string, (300, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorcode)
    elif location == "topleft":
        cv2.putText(frame, string, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorcode)
    elif location == "topright":
        cv2.putText(frame, string, (995, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorcode)
    elif location == "bottomleft":
        cv2.putText(frame, string, (5, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorcode)
    elif location == "bottomright":
        cv2.putText(frame, string, (995, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorcode)
    elif location == "top":
        cv2.putText(frame, string, (500, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colorcode)
    else:
        print "Error with onLayMessage, unknown location: ",location



def show_webcam(photos,duration):
    TRAINSET = "lbpcascade_frontalface.xml"
    DOWNSCALE = 4
    webcam = cv2.VideoCapture(0)
    cv2.namedWindow("preview")
    classifier = cv2.CascadeClassifier(TRAINSET)
    images=[]

    if webcam.isOpened(): # try to get the first frame
        rval, frame = webcam.read()
    else:
        rval = False


    n=1
    photos = int(photos)
    duration = float(duration)
    interval = duration/photos
    train,trecord = False, False
    validate,vrecord = False,False

    while rval:
    # detect faces and draw bounding boxes
        minisize = (frame.shape[1]/DOWNSCALE,frame.shape[0]/DOWNSCALE)
        miniframe = cv2.resize(frame, minisize)
        faces = classifier.detectMultiScale(miniframe)
        key = cv2.waitKey(20)
        for f in faces:
            x, y, w, h = [ v*DOWNSCALE for v in f ]
            #cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
        if key == ord('t'):
            starttime= time.time()
            train=True
            trecord = True

        if key == ord('v'):
            starttime = time.time()
            validate = True
            vrecord = True
            

        if vrecord:
            onLayMessage("REC FOR VALIDATION","top",frame,"blue")

        if trecord:
            onLayMessage("REC FOR TRAIN","top",frame,"red")

        if trecord or vrecord:
            currtime = time.time()
            timediff = currtime-starttime
            if (timediff > (interval*n)):
                roi = frame[y:y+h,x:x+w]
                images.append(frame)
                n+=1
            if n >= photos+1 :
                trecord,vrecord=False,False
                break


        cv2.putText(frame, "Press ESC to close.", (5, 25),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
        onLayMessage("Press t to train, v to validate","bottom",frame)
        cv2.imshow("preview", frame)
        # get next frame
        rval, frame = webcam.read()
        if key in [27, ord('Q'), ord('q')]: # exit on ESC
            frame[y:y+h,x:x+w]

    if validate == True:
        print "validataeee"
        return images, 'validate'
    elif train == True:
        print "trainnnneee"
        return images, 'train'
    else:
        print "v",validate
        print "t",train
        print "something wrong happened"
        return 0
            

def store(images,name,normalize,option):


    if os.path.isdir(option):
        os.chdir(option)
    else:
        os.makedirs(option)
        os.chdir(option)

        
    labels = open('labels.txt','w+')
    filepath = os.getcwd()
    for image in images:
        if normalize:
            normalizer.normalize_and_store(image,name+str(time.time())+'.jpg')
            labels.write(str(filepath)+'/'+name+str(time.time())+'.jpg\n')
        else:
            normalizer.store(image,name+str(time.time())+'.jpg')
            labels.write(str(filepath)+'/'+name+str(time.time())+'.jpg\n')


def main(filename,photos,duration,normalize):
    images,option = show_webcam(photos,duration)
    store(images,filename,normalize,option)




main(sys.argv[1],5,5,True)






