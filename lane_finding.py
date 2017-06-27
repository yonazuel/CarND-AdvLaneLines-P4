import numpy as np
import cv2
import glob
from moviepy.editor import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# First, I want to calibrate the camera to be able to undistort the images I get from the camera
# before analyzing it. Indeed, because of the lenses of the camera, the pictures it takes
# represent a distorted view of the reality:
# distances and shapes might not be represented correctly.
# So, to perform a correct analysis of the image, I first need to undistort it.
# To do that, I need to know the distortion this particular camera applies to the pictures it takes.
# I am going to measure this distortion thanks to the chessboard pictures in the camera_cal folder.
def camera_calibration():
	# I retrieve the names of the pictures to use in the camer_cal folder
	image_names = glob.glob('camera_cal/calibration*.jpg')
	
	# The chessboards in these pictures have 9 corners per row and 6 per column
	nx = 9
	ny = 6
	
	# I prepare my object points: these represents real 3D points
	# objp are the coordinates of the corners of the chessboard in the real world
	# Since the chessboard lays on a flat surface, the 3rd coordinate will always be 0
	objp = np.zeros((nx*ny,3), np.float32)
	objp[:,:2] = np.mgrid[:nx,:ny].T.reshape(-1,2)
	
	# I am going to store the coordinates of the corners in these 2 arrays
	objpoints = [] # Here the 3d coordinates in the real world
	imgpoints = [] # Here the 2d coordinates in the distorted image

	# I am going through all the calibration images
	for i in range(len(image_names)):
		# I load the image and convert it to grayscale
		img = cv2.imread(image_names[i])
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# I detect the coordinates of the corners in the distorted image
		# ret is True if the corners are found and False otherwise
		ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

		# When the corners are found, I store their coordinates in the image in imgpoints
		# I also store the coordinates of the corner in the real world in objpoints (these never change)
		if ret:
			objpoints.append(objp)
			imgpoints.append(corners)

	image_size = (img.shape[1], img.shape[0])

	# This is where the calibration of the camera is performed
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

	# I return the calibration matrix and the distortion parameters
	return mtx, dist

# Once the calibration of the camera is done and I have mtx and dist,
# I can undistort any image taken by the same camera
def undistort_image(image, mtx, dist):
	return cv2.undistort(image, mtx, dist, None, mtx)


# Once the image has been distorted, I can start analysing it.
# The goal is to find the pixels in the image that correspond to the lines.
# So, first, I am going to threshold the image. This means that I am going to eliminate
# all the pixels that don't satisfy certain conditions 
# in order to have almost only the lines left on the pictures.
# 
# The first thresholding will be regarding the gradient:
# this function will give back a binary image where only the pixels where the gradient
# in one direction (orient) is in a certain range (abs_thresh) will be present.
def abs_sobel_threshold(image, orient='x', abs_thresh=(0,255)):
	# I convert the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# I take the gradient in the specified direction
	if orient == 'x':
		sobel = cv2.Sobel(gray,cv2.CV_64F, 1, 0)
	elif orient == 'y':
		sobel = cv2.Sobel(gray,cv2.CV_64F, 0, 1)

	# I take the absolute value of this gradiant and normalize it between 0 and 255
	abs_sobel = np.absolute(sobel)
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

	# I create an empty binary image of the same size
	s_bin = np.zeros_like(scaled_sobel)
	# I set to 1 (white) only the pixels where the normalized absolute gradient
	# is in the range abs_thresh
	s_bin[(scaled_sobel>=abs_thresh[0]) & (scaled_sobel<=abs_thresh[1])] = 1

	return s_bin

# I am going to apply the exact same principle but by thresholding on the magnitude of the gradient
# This magnitude is defined by sqrt(grad_x**2 + grad_y**2)
def mag_threshold(image, sobel_kernel=3, mag_thresh=(0, 255)):
	# I convert the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# I take the gradient in both directions
	sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

	# I compute the gradient magnitude and normalize it between 0 and 255
	grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
	grad_mag = np.uint8(255*grad_mag/np.max(grad_mag)) 

	# I create an empty binary image of the same size
	gm_bin = np.zeros_like(grad_mag)
	# I set to 1 (white) only the pixels where the normalized magnitude of the gradient
	# is in the range mag_thresh
	gm_bin[(grad_mag>=mag_thresh[0]) & (grad_mag<=mag_thresh[1])] = 1

	return gm_bin

# I am going to apply the exact same principle but by thresholding on the direction of the gradient
# This direction can be computed as arctan(grad_y/grad_x)
def dir_threshold(image, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
	# I convert the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# I take the gradient in both directions
	sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

	# I compute the gradient magnitude
	# Since I compute the direction on the absolute values, 
	#the result is going to be between 0 and Pi/2
	grad_dir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))

	# I create an empty binary image of the same size
	dir_bin =  np.zeros_like(grad_dir).astype(np.uint8)
	# I set to 1 (white) only the pixels where the direction of the gradient
	# is in the range dir_thresh
	dir_bin[(grad_dir>=dir_thresh[0]) & (grad_dir<=dir_thresh[1])] = 1

	return dir_bin

# Another type of thresholding I use is regarding the color.
# The same principle applies, but this time I apply thresholding on the 3 color channels separately
def color_threshold(image, color_space='hsv', ch1_thresh=(0,255), ch2_thresh=(0,255), ch3_thresh=(0,255)):
	# I convert the image to the specified color space
	if color_space == 'hls':
		color_img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	elif color_space == 'hsv':
		color_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	else:
		color_img = np.copy(image)

	# I separate the 3 color channels
	ch1 = color_img[:,:,0]
	ch2 = color_img[:,:,1]
	ch3 = color_img[:,:,2]

	# I create an empty binary image of the same size
	color_bin = np.zeros_like(ch1).astype(np.uint8)
	# I set to 1 (white) only the pixels where all the color channels are in their authorized range
	color_bin[(ch1>=ch1_thresh[0]) & (ch2>=ch2_thresh[0]) & (ch3>=ch3_thresh[0]) & \
			  (ch1<=ch1_thresh[1]) & (ch2<=ch2_thresh[1]) & (ch3<=ch3_thresh[1])] = 1

	return color_bin

# Now I need to combine all these thresholding methods to obtain one binary image
# where almost only the line pixels are white.
def combines_threshold(image, abs_thresh_x=(25,255), abs_thresh_y=(25,255), \
						  color_space='hsv', ch1_thresh=(0,255), ch2_thresh=(80,255), ch3_thresh=(80,255)):
	
	# I decide to threshold according to the 2 gradients
	s_bin_x = abs_sobel_threshold(image, 'x', abs_thresh_x)
	s_bin_y = abs_sobel_threshold(image, 'y', abs_thresh_y)
	# I also decide to threshold according to the S and V channels 
	# (since ch1_thresh=(0,255) and all pixels have values between 0 and 255,
	# the result does not depend on the H channel)
	color_bin = color_threshold(image, color_space, ch1_thresh, ch2_thresh, ch3_thresh)

	# I create an empty binary image of the same size
	combined_bin = np.zeros_like(color_bin).astype(np.uint8)
	# I set to 1 (white) only the pixels where either the gradients pass the threshold
	# or the color passes the threshold.
	combined_bin[((s_bin_x==1) & (s_bin_y==1)) | (color_bin==1)] = 1

	return combined_bin

# Now that I have obtained binary images where almost only the lines are nonzero pixels,
# I want to clearly identify the lines.
# To make this identification easier, I am first going to change the perspective:
# I want to have a view of the road from above instead of from the center of the windshield.
# To perform this change of perspective, I first need to compute the transition matrix
# I am also computing the inverse of this matrix to be able to come back from an above view
# to a windshield view.
# To compute these matrices, I need to map 4 points on the original (undistorted) image
# to 4 new points.
# Since I want to find the lines on the road, I chose to map a trapezoidal region on the original image
# to a rectangular region on the new image.
# I picked the coordinates of the 4 corners of the trapezoidal region on the 2 test images
# straight_lines1 and straight_lines2 after I applied undistortion
# (I chose not to include the hood of the car in this region)
def get_perspective_matrices(image_size=(1280,720), offset=0.15):
	# These are the dimensions of the original (undistorted) image
	x_shape = image_size[0]
	y_shape = image_size[1]

	# These are the coordinates of the 4 corners of the original trapezoidal region
	#c1_src = [585, 460]
	#c2_src = [695, 460]
	#c3_src = [1127, 720]
	#c4_src = [203, 720]
	c1_src = [584, 460]
	c2_src = [700, 460]
	c3_src = [1008, 660]
	c4_src = [300, 660]

	# These are the coordinates of the 4 corners in the new image
	c1_dst = [x_shape*offset, 0]
	c2_dst = [x_shape*(1-offset), 0]
	c3_dst = [x_shape*(1-offset), y_shape]
	c4_dst = [x_shape*offset, y_shape]

	src = np.float32([c1_src, c2_src, c3_src, c4_src])
	dst = np.float32([c1_dst, c2_dst, c3_dst, c4_dst])

	# Now I compute the two transition matrices
	M = cv2.getPerspectiveTransform(src,dst)
	Minv = cv2.getPerspectiveTransform(dst,src)

	return M, Minv

# So, to change perspective, I will just need to use this function
def change_perspective(undist_image, M):
	image_size = (undist_image.shape[1], undist_image.shape[0])
	return cv2.warpPerspective(undist_image, M, image_size, flags=cv2.INTER_LINEAR)

# Now that I have undistorted the image, thresholded it and change my perspective to an above view,
# I can look for the lines.
# There are two ways to look for the lines. 
# Either you have no idea where they are, or you do and can look arround where you think they are.
# First let's implement a function that searches for the lines not knowing where they might be.
# We use a histogram to look for the starting point of the lines 
# and then apply a sliding window method to determine the pixels belonging to the lines.
def find_lane_from_scratch(image, nb_windows=20):
	# I define the dimensions of the image
	# This is a binary warped image (view from above)
	x_shape = image.shape[1]
	y_shape = image.shape[0]

	# I take a histogram along the columns of the bottom half of the image
	# This will give me the number of pixels in each column that are nonzero
	# The columns with the maximum of nonzero pixels will be the starting points of my lines
	histogram = np.sum(image[y_shape/2:,:], axis=0)

	# I know one line is on the left and the other on the right.
	# So I am going to look at the peak of my histogram on the left and right
	# of the middle of the image
	mid_point = np.int(histogram.shape[0]/2)
	x_left = np.argmax(histogram[:mid_point])
	x_right = np.argmax(histogram[mid_point:]) + mid_point

	# I set the height and width of each window
	window_height = np.int(y_shape/nb_windows)
	window_width = 100

	# I identify the x and y positions with nonzero pixels in the image
	nonzero = image.nonzero()
	x_nonzero = np.array(nonzero[1])
	y_nonzero = np.array(nonzero[0])

	# At each step (going from a window to the next), I want to recenter my window arround the
	# new maximum of nonzero pixels found. But I am going to recenter this window only if 
	# the number of nonzero pixels is at least min_pix
	min_pix = int(window_height*window_width/20)

	# I will store the indices of the pixels that I consider being part of the lines
	# in those lists
	left_line_inds = []
	right_line_inds = []

	for window in range(nb_windows):
		# The coordinates of the edges of the windows
		y_low = y_shape - (window+1)*window_height
		y_high = y_low + window_height
		x_left_low = x_left - window_width
		x_left_high = x_left + window_width
		x_right_low = x_right - window_width
		x_right_high = x_right + window_width

		# I identify the indices of the nonzero pixels in the current windows
		current_left_indices = ((y_nonzero >= y_low) & (y_nonzero < y_high) & (x_nonzero >= x_left_low) & (x_nonzero < x_left_high)).nonzero()[0]
		current_right_indices = ((y_nonzero >= y_low) & (y_nonzero < y_high) & (x_nonzero >= x_right_low) & (x_nonzero < x_right_high)).nonzero()[0]

		# I add them to the list of pixels that I consider part of the lines
		left_line_inds.append(current_left_indices)
		right_line_inds.append(current_right_indices)

		# If I found enough nonzero pixel in a window, I recenter the next window arround
		# the average position of the nonzero pixels found
		if len(current_left_indices) > min_pix:
			x_left = np.int(np.mean(x_nonzero[current_left_indices]))
		if len(current_right_indices) > min_pix:        
			x_right = np.int(np.mean(x_nonzero[current_right_indices]))

	# I flatten the lists of pixels belonging to the lines
	left_line_inds = np.concatenate(left_line_inds)
	right_line_inds = np.concatenate(right_line_inds)

	# I extract the coordinates of the pixels of the right and left lines
	x_left_line = x_nonzero[left_line_inds]
	y_left_line = y_nonzero[left_line_inds] 
	x_right_line = x_nonzero[right_line_inds]
	y_right_line = y_nonzero[right_line_inds] 

	# This gives me the 2 lines and the lane
	left_line = [x_left_line, y_left_line]
	right_line = [x_right_line, y_right_line]
	lane = [left_line, right_line]

	return lane

# Since I want to draw smooth lines, I am going to fit a 2nd degree polynomial
def fit_poly(line):
	return np.polyfit(line[1], line[0], 2)

# Then, to evaluate this polynomial in a point
def eval_poly(poly, y):
	return poly[0]*(y**2) + poly[1]*y + poly[2]

# When analysing a video frame by frame, to gain some time, I am not going in each frame 
# to try to find the lines from scratch. I am going to use the position of the lines in the
# previous frame to find them in this frame.
def find_lane_from_previous(image, previous_lane):
	# I identify the x and y positions with nonzero pixels in the image
	nonzero = image.nonzero()
	y_nonzero = np.array(nonzero[0])
	x_nonzero = np.array(nonzero[1])
	
	# I set the width of the window
	window_width = 100

	# I fit 2 polynomials to the previous right and left lines
	left_fit = fit_poly(previous_lane[0])
	right_fit = fit_poly(previous_lane[1])

	# I define where I am going to look for my new lines:
	# I am looking 100 pixels to the right and left of where the lines were before
	x_left = eval_poly(left_fit, y_nonzero)
	x_left_low = x_left - window_width
	x_left_high = x_left + window_width

	x_right = eval_poly(right_fit, y_nonzero)
	x_right_low = x_right - window_width
	x_right_high = x_right + window_width

	# I take the indices of the nonzero pixels arround the previous lines
	left_line_inds = ((x_nonzero >= x_left_low) & (x_nonzero <= x_left_high))
	right_line_inds = ((x_nonzero >= x_right_low) & (x_nonzero <= x_right_high))

	# I extract the coordinates of the pixels of the right and left lines
	x_left_line = x_nonzero[left_line_inds]
	y_left_line = y_nonzero[left_line_inds] 
	x_right_line = x_nonzero[right_line_inds]
	y_right_line = y_nonzero[right_line_inds]

	# This gives me the 2 lines and the lane
	left_line = [x_left_line, y_left_line]
	right_line = [x_right_line, y_right_line]
	lane = [left_line, right_line]

	return lane

# The question now is to know when to use the lines of the previous frame 
# to find the lines in this frame and when to try to find them from scratch.
# For that, I am going to check if the lines already found are correct.
# If I am sure they are correct (proba_correct>50%), I can use them to find the lines
# in the next frame. Otherwise, I will have to try to find new lanes from scratch.
# This check is in 3 parts:
# - first I check that the right and left lines have approximatively the same curvature
# - then I check that they are parallel
# - finally, I check that the lane is wide of about 3.7 meters.

# For the first check, I need to compute the curvature.
# I will compute this curvature in meters (real world) and not pixels.
def curvature(line, image_size=(1280,720), offset=0.15):
	# I define the conversion ratios in x and y from pixels space to meters (real world)
	x_m_per_pix = 3.7/(image_size[0]*(1-2*offset)) # meters per pixel in x dimension
	y_m_per_pix = 30/image_size[1] # meters per pixel in y dimension
	
	# I fit a polynomial to the real line
	fit_real_line = np.polyfit(line[1]*y_m_per_pix, line[0]*x_m_per_pix, 2)
	
	# I want to compute the curvature at the bottom of the image (where the car is)
	y_eval = image_size[1]*y_m_per_pix
	
	# So the radius of curvature is
	r_curv = ((1 + (2*fit_real_line[0]*y_eval + fit_real_line[1])**2)**1.5) / np.absolute(2*fit_real_line[0])
	
	return r_curv

# To check if the lanes are parallel, I check if the global slope is the same
def slope(line, image_size=(1280,720)):
	# I fit the polynomial (for this check, it does not really matter if I do
	# it in the real world or the pixel space)
	fit_line = fit_poly(line)

	# I evaluate it at the top and bottom of the image
	x_top = eval_poly(fit_line, 0)
	x_bottom = eval_poly(fit_line, image_size[1])

	# So the global slope is
	sl = (x_top-x_bottom)/image_size[1] 
	
	return sl

# For the last check, I need to compute the width of the lane in the real world
# So I fit poynomial to both lanes and evaluate them at the bottom of the image
# Then I take the difference
def lane_width(lane, image_size=(1280,720), offset=0.15):
	# I define the conversion ratios in x and y from pixels space to meters (real world)
	x_m_per_pix = 3.7/(image_size[0]*(1-2*offset))
	y_m_per_pix = 30/image_size[1]
	
	# I fit a polynomial to the real lines
	left_line = lane[0]
	fit_left_line = np.polyfit(left_line[1]*y_m_per_pix, left_line[0]*x_m_per_pix, 2)
	right_line = lane[1]
	fit_right_line = np.polyfit(right_line[1]*y_m_per_pix, right_line[0]*x_m_per_pix, 2)
	
	# I want to compute the curvature at the bottom of the image (where the car is)
	y_eval = image_size[1]*y_m_per_pix

	# So the width of the lane is
	width = eval_poly(fit_right_line,y_eval) - eval_poly(fit_left_line,y_eval) 

	return width

# Then, my check on the probability that the lines are correct is
def check_lane_sanity(lane, image_size=(1280,720), offset=0.15):
	left_line = lane[0]
	right_line = lane[1]
	
	# I compute the curvatures of the 2 lines and compare them
	left_curvature = curvature(left_line, image_size, offset)
	right_curvature = curvature(right_line, image_size, offset)
	pr_correct_curvature = 1 - np.abs(left_curvature-right_curvature)/np.abs(left_curvature)

	# I compute the slopes of the 2 lines and compare them
	left_slope = slope(left_line, image_size)
	right_slope = slope(right_line, image_size)
	pr_correct_slope = 1 - np.abs(left_slope-right_slope)/np.abs(left_slope)

	# I compute the width of the lane and compare it to 3.7 meters
	width = lane_width(lane, image_size, offset)
	pr_correct_width = 1 - np.abs(width-3.7)/3.7

	# I compute the final probability that the lines are correct
	pr_correct = 1./3 * (pr_correct_curvature + pr_correct_slope + pr_correct_width)

	return pr_correct

# On the final image, I will write the curvature of the lane, but also the distance
# of the car to the center of the lane.
# To compute this distance, I consider the center of the lane to be the middle between
# the 2 points at the basis of the lines and the position of the car to be the middle
# of the image.
def distance_to_center(lane, image_size=(1280,720), offset=0.15):
	left_line = lane[0]
	right_line = lane[1]

	# I fit polynomials to the lines
	fit_left_line = fit_poly(left_line)
	fit_right_line = fit_poly(right_line)

	# I evaluate the polynomials at the bottom of the image (where the car is)
	y_eval = image_size[1]
	x_left = eval_poly(fit_left_line, y_eval)
	x_right = eval_poly(fit_right_line, y_eval)

	# Then I take the midpoint and the difference with the center of the image
	middle_road = 1./2 * (x_left + x_right)
	diff_center = abs(middle_road - image_size[0]/2)

	# I now need to convert it to meters
	x_m_per_pix = 3.7/(image_size[0]*(1-2*offset))
	diff_center_real = diff_center * x_m_per_pix

	return diff_center_real

# Now I can finally detect the lane on an image (undistorted, thresholded and viewed from above)
# Since the goal is to do it in real time, I am going to find the lines on the image
# (if the probability of correctness of the previous lines is high enough I do't do it from scratch)
# But if the lines I found are not correct, I don't use them, I just use the previous ones and
# go to the next frame (where I will be force to find new lines from scratch) instead of reusing this
# frame and try to find the lanes from scratch
def detect_lane(image, previous_lane, previous_proba, proba_thresh=0.5, offset=0.15):
	image_size = (image.shape[1], image.shape[0])
	
	# I check the probability of correctness of the previous lane to see if I have
	# to find the lines from scratch or arround the previous ones
	if previous_proba > proba_thresh:
		lane = find_lane_from_previous(image, previous_lane)
	else:
		lane = find_lane_from_scratch(image)

	# Now that I found the lines, I want to check if they are correct.
	# If they are not, I use the previous ones and go to the next frame
	current_proba = check_lane_sanity(lane, image_size, offset)

	if previous_proba > proba_thresh and current_proba <= proba_thresh:
		return previous_lane, current_proba
	else:
		return lane, current_proba

# Now I am going to draw the lane, and print the curvature on the image
def draw_lane(undist_image, bin_image, lane, Minv):
	# First I create an image to draw the lane on
	warp_zero = np.zeros_like(bin_image).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# I extract the lines
	left_line = lane[0]
	right_line = lane[1]

	# I fit a polynomial to each of them
	left_fit = fit_poly(left_line)
	right_fit = fit_poly(right_line)

	# I want to draw on all the length of the image (the one with the view from above)
	plot_y = np.linspace(0, 719, num=720)
	plot_left_x = eval_poly(left_fit, plot_y)
	plot_right_x = eval_poly(right_fit, plot_y)

	# So the points of the lane are
	pts_left = np.array([np.transpose(np.vstack([plot_left_x, plot_y]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([plot_right_x, plot_y])))])
	pts = np.hstack((pts_left, pts_right))

	# I color those points in green
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# I change perspective to go back to dashboard view
	driver_perspective = change_perspective(color_warp, Minv)

	# I add the colored lane to the original (undistorted) image and add the curvature
	final_image = cv2.addWeighted(undist_image, 1, driver_perspective, 0.3, 0)
	
	curv = round(1./2*(curvature(left_line)+curvature(right_line)),1)
	dist_to_center = round(distance_to_center(lane),2)
	
	cv2.putText(final_image, 'Curvature: '+str(curv)+' m', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	cv2.putText(final_image, 'Distance to center: '+str(dist_to_center)+' m', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

	return final_image

# To avoid jittery lane, I smooth the drawing by using an average with the previous lanes
def draw_lane_smooth(undist_image, bin_image, lane, previous_lanes, Minv):
	# First I create an image to draw the lane on
	warp_zero = np.zeros_like(bin_image).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# I extract the lines
	left_line = lane[0]
	right_line = lane[1]

	# I add the previous lines
	for pr_lane in previous_lanes:
		left_line = np.append(left_line,pr_lane[0], axis=1)
		right_line = np.append(right_line,pr_lane[1], axis=1)

	# I fit a polynomial to each of them
	left_fit = fit_poly(left_line)
	right_fit = fit_poly(right_line)

	# I want to draw on all the length of the image (the one with the view from above)
	plot_y = np.linspace(0, 719, num=720)
	plot_left_x = eval_poly(left_fit, plot_y)
	plot_right_x = eval_poly(right_fit, plot_y)

	# So the points of the lane are
	pts_left = np.array([np.transpose(np.vstack([plot_left_x, plot_y]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([plot_right_x, plot_y])))])
	pts = np.hstack((pts_left, pts_right))

	# I color those points in green
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# I change perspective to go back to dashboard view
	driver_perspective = change_perspective(color_warp, Minv)

	# I add the colored lane to the original (undistorted) image and add the curvature
	final_image = cv2.addWeighted(undist_image, 1, driver_perspective, 0.3, 0)
	
	curv = round(1./2*(curvature(left_line)+curvature(right_line)),1)
	dist_to_center = round(distance_to_center(lane),2)
	
	cv2.putText(final_image, 'Curvature: '+str(curv)+' m', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	cv2.putText(final_image, 'Distance to center: '+str(dist_to_center)+' m', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

	return final_image

# So, putting all of this together, each frame will be processed by:
# - undistorting the image
# - thresholding the image
# - changing perspective
# - detecting the lane
# - drawing the lane and going back to original perspective
def process_image(image, mtx, dist, M, Minv, previous_lane=[], previous_proba=0, previous_lanes=[], nb_previous_lanes=5):
	undist_image = undistort_image(image, mtx, dist)
	bin_image = combines_threshold(undist_image)
	warp_bin_image = change_perspective(bin_image, M)
	lane, current_proba = detect_lane(warp_bin_image, previous_lane, previous_proba)
	final_image = draw_lane_smooth(undist_image, warp_bin_image, lane, previous_lanes, Minv)
	previous_lanes.append(lane)
	if len(previous_lanes)>nb_previous_lanes:
		previous_lanes.pop(0)
	return final_image, lane, current_proba, previous_lanes

# Let's apply all that to the video
def process_video(clip, mtx, dist, M, Minv):
	# First I initialize my parameters
	previous_proba = 0
	previous_lane = []
	previous_lanes = []
	
	new_frames = []
	counter = 1
	nb_frames = clip.fps * clip.duration

	# Then I go through each frame and process it
	for frame in clip.iter_frames():
		new_frame, lane, proba, previous_lanes = process_image(frame, mtx, dist, M, Minv, previous_lane, previous_proba, previous_lanes)
		new_frames.append(new_frame)
		previous_lane = lane
		previous_proba = proba
		print('Processing image: ' + str(counter) + '/' + str(nb_frames), end='\r')
		counter = counter + 1

	print('')
	new_clip = ImageSequenceClip(new_frames, fps=clip.fps)

	return new_clip

mtx, dist = camera_calibration()
M, Minv = get_perspective_matrices()

clip = VideoFileClip('project_video.mp4')
process_clip = process_video(clip, mtx, dist, M, Minv)
process_clip.write_videofile('project_video_out.mp4')


