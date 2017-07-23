import cv2
import numpy as np
import image_analysis

class DrawFrame:
	def __init__(self):				
		self.row_ratio = None
		self.col_ratio = None
		self.text = ''
		self.box_size = 40
		self.hand_row_nw = None
		self.hand_row_se = None
		self.hand_col_nw = None
		self.hand_col_se = None
		self.pw = ""
		

	def resize(self, frame):
		rows,cols,_ = frame.shape
		
		ratio = float(cols)/float(rows)
		new_rows = 400
		new_cols = int(ratio*new_rows)
		
		self.row_ratio = float(rows)/float(new_rows)
		self.col_ratio = float(cols)/float(new_cols)
		
		resized = cv2.resize(frame, (new_cols, new_rows))	
		return resized


	def flip(self, frame):
		flipped = cv2.flip(frame, 1)
		return flipped	


	def draw_final(self, frame, paper_detection, hand_detection):
		hand_masked = image_analysis.apply_hist_mask(frame, hand_detection.hand_hist)
		# paper_hand = paper_detection.paper_copy()
		# self.plot_word_boxes(paper_hand, paper_detection.words)

		contours = image_analysis.contours(hand_masked)
		if contours is not None and len(contours) > 0:
			max_contour = image_analysis.max_contour(contours)
			hull = image_analysis.hull(max_contour)
			centroid = image_analysis.centroid(max_contour)
			defects = image_analysis.defects(max_contour)

			if centroid is not None and defects is not None and len(defects) > 0:	
				farthest_point = image_analysis.farthest_point(defects, max_contour, centroid)

				if farthest_point is not None:
					self.plot_farthest_point(frame, farthest_point)
					self.plot_hull(frame, hull)

					point = self.original_point(farthest_point)
					self.farthest_point = farthest_point
					# print(farthest_point)
					x, y = farthest_point
					y -= 400

					# a,b are the top-left coordinate of the rectangle and (c,d) be its width and height.
					# to judge a point(x0,y0) is in the rectangle, just to check
					# if a < x0 < a+c and b < y0 < b + d

					for i in range(9):
						#if farthest_point is in any of the rectangles
						#then we add that to string
						#self.hand_col_nw[i], self.hand_row_nw[i] a, b
						#self.hand_col_se[i],self.hand_row_se[i] c, d

						# # if i == 0:
						# print(i)
						# print(self.hand_col_nw[i], self.hand_row_nw[i])
						# print(self.hand_col_se[i],self.hand_row_se[i])
						# print(x, y)
						# print("=================")

						if self.hand_col_nw[i] < x and x < self.hand_col_se[i] and self.hand_row_nw[i] < y and y < self.hand_row_se[i]:
							print self.pw, i
							if len(self.pw) < 1 or self.pw[-1] != str(i):
								self.pw += str(i)
		
		frame_final = frame
		return frame_final


	def original_point(self, point):
		x,y = point
		xo = int(x*self.col_ratio)
		yo = int(y*self.row_ratio)
		return (xo,yo)


	def new_point(self, point):
		(x,y) = point
		xn = int(x/self.col_ratio)
		yn = int(y/self.row_ratio)
		return (xn,yn)

	
	def plot_defects(self, frame, defects, contour):
		if len(defects) > 0:
			for i in xrange(defects.shape[0]):
				s,e,f,d = defects[i,0]
				start = tuple(contour[s][0])
				end = tuple(contour[e][0])
				far = tuple(contour[f][0])               
				cv2.circle(frame, start, 5, [255,0,255], -1)


	def plot_farthest_point(self, frame, point):
		cv2.circle(frame, point, 5, [0,0,255], -1)			

	
	def plot_centroid(self, frame, point):
		cv2.circle(frame, point, 5, [255,0,0], -1)

	
	def plot_hull(self, frame, hull):
		cv2.drawContours(frame, [hull], 0, (255,0,0), 2)	


	def plot_contours(self, frame, contours):
		cv2.drawContours(frame, contours, -1, (0,255,0), 3)				


	def plot_text(self, frame, text):
		cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_PLAIN, 3, [255,255,255], 4)

	def plot_word_boxes(self, frame, words):
		rows,cols,_ = frame.shape
		for w in words:
			x_nw,y_nw,x_se,y_se = w.box
			x_nw,y_nw = self.new_point((x_nw,y_nw))
			x_se,y_se = self.new_point((x_se,y_se))
			x_nw = x_nw
			x_se = x_se

			cv2.rectangle(frame,(x_nw,y_nw),(x_se,y_se),
									(0,255,255),1)	

	def draw_pass_rect(self, frame):
		rows,cols,_ = frame.shape
		
		div = 20
		row1 = 6 #6
		row2 = 10 #10
		row3 = 14 #14
		self.hand_row_nw = np.array([row1*rows/div,row1*rows/div,row1*rows/div, #6
														row2*rows/div,row2*rows/div,row2*rows/div, #10
														row3*rows/div,row3*rows/div,row3*rows/div])#14
		col1 = 8
		col2 = 10
		col3 = 12
		self.hand_col_nw = np.array([col1*cols/div,col2*cols/div,col3*cols/div,
														col1*cols/div,col2*cols/div,col3*cols/div,
														col1*cols/div,col2*cols/div,col3*cols/div])
		self.hand_row_se = self.hand_row_nw + self.box_size #size of box
		self.hand_col_se = self.hand_col_nw + self.box_size

		size = self.hand_row_nw.size
		for i in range(size):
			cv2.rectangle(frame,(self.hand_col_nw[i],self.hand_row_nw[i]),(self.hand_col_se[i],self.hand_row_se[i]),
										(0,255,0),1)
			cv2.putText(frame, str(i)+"", ((self.hand_col_nw[i])+15,(self.hand_row_nw[i])+25), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2);
		black = np.zeros(frame.shape, dtype=frame.dtype)
		frame_final = np.vstack([black, frame])
		return frame_final
