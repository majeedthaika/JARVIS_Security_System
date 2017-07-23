import cv2
import sys
import time
import normalizer
import os
import FR_NN_train
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
	cv2.namedWindow("train")
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
	train = False

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

		if not train:
			starttime= time.time()
			train = True

		if train:
			currtime = time.time()
			timediff = currtime-starttime
			if (timediff > (interval*n)):
				roi = frame[y:y+h,x:x+w]
				images.append(frame)
				n+=1
			if n >= photos:
				break

		cv2.imshow("train", frame)
		# get next frame
		rval, frame = webcam.read()

	return images, 'train'
			

def store(images,name,name_id,normalize,option):
	if not os.path.isdir(option):
		os.makedirs(option)

	labels = open(option+'/'+'labels.txt','a+')
	for image in images:
		if normalize:
			normalizer.normalize_and_store(image,option+"/"+name+"_"+str(time.time())+'.jpg')
		else:
			normalizer.store(image,option+"/"+name+"_"+str(time.time())+'.jpg')

		labels.write(str(os.getcwd())+"/"+option+'/'+name+"_"+str(time.time())+'.jpg '+str(name_id)+'\n')

def find_name_id(name):
	name_id = 0
	if (os.path.isfile('name_id.txt')):
		with open('name_id.txt','r+') as f:
			for line in f:
				if (name == "".join(line.split())):
					return name_id
				else:
					name_id+=1

	with open('name_id.txt','a+') as f:
		f.write(name+"\n")

	return name_id

def main(name,photos,duration,normalize):
	name_id = find_name_id(name)
	images,option = show_webcam(photos,duration)
	store(images,name,name_id,normalize,option)
	train_NN()

main(sys.argv[1],10,5,True)






