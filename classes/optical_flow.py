import sys
import cv2
import numpy as np

# taken directly from here: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

class optical_flow:
	feature_params	= None
	lk_params	= None
	color		= None
	mask		= None
	p0		= None
	old_gray	= None

	fail_counter	= 0
	max_fail	= 10

	debug		= False

	def __init__(self, first_frame, debug=False):
		self.debug = debug

		# params for ShiTomasi corner detection
		self.feature_params = dict(maxCorners = 100,
		                       qualityLevel = 0.3,
		                       minDistance = 7,
		                       blockSize = 7)
		
		# Parameters for lucas kanade optical flow
		self.lk_params = dict( winSize  = (15, 15),
		                  maxLevel = 2,
		                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
		
		# Create some random colors
		self.color = np.random.randint(0, 255, (100, 3))

		if self.debug:
			print(f'{first_frame.shape = } - {first_frame[:100] = }')

		# Take first frame and find corners in it
		#ret, old_frame = cap.read()
		self.old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
		self.p0 = cv2.goodFeaturesToTrack(self.old_gray, **self.feature_params, mask=None)
		if self.debug:
			print(f'{self.p0 = }')
	
		# Create a mask image for drawing purposes
		self.mask = np.zeros_like(first_frame)

		if self.debug:
			print('Optical flow successfully initialized...')

		self.fail_counter = 0
	
		return
	
	def do_opt_flow(self, frame):
		if self.debug:
			print(f'{frame.shape = } - {frame[:100] = }')
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# calculate optical flow
		p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

		if self.debug:
			print(f'{p1 = }')
			print(f'{st = }')
			print(f'{err = }')
			#sys.exit(0)
	
		# Select good points
		if p1 is not None:
			good_new = p1[st==1]
			good_old = self.p0[st==1]
		else:
			if self.debug:
				print(f'No optical flow has been performed at this iteration - {self.fail_counter = }')

			self.fail_counter += 1
			if self.fail_counter >= self.max_fail:
				self.__init__(frame, debug=self.debug)

			return None, None
	
		# draw the tracks
		for i, (new, old) in enumerate(zip(good_new, good_old)):
			a, b = new.ravel()
			c, d = old.ravel()
			self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
			img = frame.copy()
			img = cv2.circle(img, (int(a), int(b)), 5, self.color[i].tolist(), -1)
		img = cv2.add(frame, self.mask)
	
		# Now update the previous frame and previous points
		self.old_gray = frame_gray
		self.p0 = good_new.reshape(-1, 1, 2)

		return img, err
	
