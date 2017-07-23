import cv2
from draw_frame import DrawFrame
# from paper_detection import PaperDetection
from hand_detection import HandDetection
import time
import csv

def authenticate(output_video, name):
	camera = cv2.VideoCapture(0)
	fgbg = cv2.BackgroundSubtractorMOG()

	login = {}
	with open('login.csv') as f:
	  reader = csv.reader(f)
	  for row in reader:
	  	login[row[0]] = row[1]
	
	print login
	#if duplicate names, it will use the pw that is newest in csv file

	if (name not in login):
		return False

	password = login[name]
	print password
	
	df = DrawFrame()
	hd = HandDetection()

	timeout = time.time() + 5

	while True:
		# get frame
		(grabbed, frame_in) = camera.read()

		fgmask = fgbg.apply(frame_in)
		cv2.imshow('fgmask', fgmask)

		# fgmask = fgbg.apply(frame_in)
	 #    # Display the fgmask frame
	 #    cv2.imshow('fgmask', fgmask)

		# shrink frame
		frame = df.resize(frame_in)
		# flipped frame to draw on
		frame_final = df.flip(frame)
		#train hand
		if time.time() > timeout:
			if not hd.trained_hand:
				hd.train_hand(frame)
		# click q to quit 
		if cv2.waitKey(1) == ord('q') & 0xFF:
			print df.pw
		 	break	

		#if hand not trained, draw red squares
		if not hd.trained_hand:
			frame_final = hd.draw_hand_rect(frame_final)
		#if hand trained, draw 9 digits
		elif hd.trained_hand:
			frame_final = df.draw_pass_rect(frame_final)
			frame_final = df.draw_final(frame_final, 12, hd)
			if len(df.pw) > 3:
				print df.pw
				break

		# display frame	
		cv2.imshow('image', frame_final)	

	camera.release()
	cv2.destroyAllWindows()

	if (password == df.pw):
		return True
	else:
		return False

def register(output_video, name):
	camera = cv2.VideoCapture(0)
	df = DrawFrame()
	hd = HandDetection()
	timeout = time.time() + 5

	while True:
		# get frame
		(grabbed, frame_in) = camera.read()
		# shrink frame
		frame = df.resize(frame_in)
		# flipped frame to draw on
		frame_final = df.flip(frame)
		#train hand
		if time.time() > timeout:
			if not hd.trained_hand:
				hd.train_hand(frame)
		# click q to quit 
		if cv2.waitKey(1) == ord('q') & 0xFF:
			print df.pw
		 	break	

		#if hand not trained, draw red squares
		if not hd.trained_hand:
			frame_final = hd.draw_hand_rect(frame_final)
		#if hand trained, draw 9 digits
		elif hd.trained_hand:
			frame_final = df.draw_pass_rect(frame_final)
			frame_final = df.draw_final(frame_final, 12, hd)
			if len(df.pw) > 3:
				print df.pw
				fd = open('login.csv','a')
				fd.write(name+","+df.pw+"\n")
				fd.close()
				break
		# display frame	
		cv2.imshow('image', frame_final)

	camera.release()
	cv2.destroyAllWindows()

	return True