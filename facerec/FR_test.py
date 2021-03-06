import cv2
import sys
import time
import normalizer
import shutil
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
	cv2.namedWindow("test")
	classifier = cv2.CascadeClassifier(TRAINSET)
	images=[]

	if webcam.isOpened(): # try to get the first frame
		rval, frame = webcam.read()
	else:
		rval = False


	n = 0
	photos = int(photos)
	duration = float(duration)
	interval = duration/photos
	test = False

	while rval:
	# detect faces and draw bounding boxes
		minisize = (frame.shape[1]/DOWNSCALE,frame.shape[0]/DOWNSCALE)
		miniframe = cv2.resize(frame, minisize)
		faces = classifier.detectMultiScale(miniframe)
		key = cv2.waitKey(20)

		if (len(faces) > 0):
			x, y, w, h = [ v*DOWNSCALE for v in faces[0] ]
		else:
			x, y, w, h = 0, 0, frame.shape[0], frame.shape[1]
			#cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))

		if not test:
			starttime= time.time()
			test = True

		if test:
			currtime = time.time()
			timediff = currtime-starttime
			if (timediff > (interval*n)):
				roi = frame[y:y+h,x:x+w]
				images.append(frame)
				n+=1
			if n >= photos:
				break

		cv2.imshow("test", frame)
		# get next frame
		rval, frame = webcam.read()

	return images, 'test'
			

def store(images,name,normalize,option):
	if os.path.isdir(option):
		shutil.rmtree(option)
	os.makedirs(option)

	labels = open(option+'/'+'labels.txt','w+')
	for image in images:
		if normalize:
			normalizer.normalize_and_store(image,option+"/"+name+"_"+str(time.time())+'.jpg')
		else:
			normalizer.store(image,option+"/"+name+"_"+str(time.time())+'.jpg')

		labels.write(str(os.getcwd())+"/"+option+'/'+name+"_"+str(time.time())+'.jpg -1\n')

def main(photos,duration,normalize):
	images,option = show_webcam(photos,duration)
	store(images,"test_img",normalize,option)
	# return test_NN()

print main(10,5,True)






