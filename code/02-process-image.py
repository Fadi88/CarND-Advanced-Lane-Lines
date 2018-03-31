import cv2
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt

def abs_sobel(img , orient = 'x' , thresh = (0,255) ):
	#extract gray scale from input image
	gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

	if orient is 'x':
		x_dir = 1
		y_dir = 0
	else:
		x_dir = 0
		y_dir = 1

	#diff the image in the given orient. and normalize the output
	abs_sobel = np.absolute(cv2.Sobel(gray , cv2.CV_64F , x_dir , y_dir))
	sobel_norm = np.uint8(255*abs_sobel / np.max(abs_sobel))
	
	#create a binary map from the output and initilize to zeros
	binary_sobel = np.zeros_like(sobel_norm)

	#apply threshold
	binary_sobel[(sobel_norm >= thresh[0]) & (sobel_norm <= thresh[1])] = 1 
	
	return binary_sobel
	
def color_threshold(img , sthresh = (0,255) , vthresh = (0,255)):
	hls = cv2.cvtColor(img , cv2.COLOR_BGR2HLS)
	s_channel = hls[:,:,2]
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

	hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
	v_channel = hsv[:,:,2]
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

	ret = np.zeros_like(s_channel)
	ret[(s_binary == 1) & (v_binary == 1)] = 1

	return ret
	
dict_pickle = pickle.load(open( "../camera_cal_output/cal-pickle.p", "rb"))

mtx  = dict_pickle['mtx']
dist = dict_pickle['dist']

#read all images as a glob object
img_glob = glob.glob("../test_images/test*.jpg")

for img_file in img_glob:
	img = cv2.imread(img_file)
	img_undist = cv2.undistort(img , mtx , dist , None , None)

	#extract binary pixel 
	#gradx absolute sobel in the x direction thresh 12 255
	gradx = abs_sobel(img , orient = 'x' , thresh = (12,255) )

	#grady absolute sobel in the y direction thresh 25 255
	grady = abs_sobel(img , orient = 'y' , thresh = (25,255) )

	#c_binary color binary with with saturation thres 100 255 and value thresh 50 255
	c_binary = color_threshold(img , sthresh = (100,255) , vthresh = (50,255))

	combined = np.zeros_like(grady)
	combined[(gradx == 1) & (grady == 1) | (c_binary == 1)] = 255
	'''
	cv2.imwrite(img_file.replace("test_images/test" , "output_images/test_x_") , gradx * 255)
	cv2.imwrite(img_file.replace("test_images/test" , "output_images/test_y_") , grady * 255)
	cv2.imwrite(img_file.replace("test_images/test" , "output_images/test_c_") , c_binary * 255)
	cv2.imwrite(img_file.replace("test_images/test" , "output_images/test_a_") , combined * 255)
	'''

	#get src and dst to prepare for prespective transform 
	img_size = (img.shape[1] , img.shape[0])
	top_off = 0.64
	bot_off = 0.9
	center_top_off = 0.018
	center_bot_off = 0.225

	x_right_bot_corr = 120
	x_right_top_corr = 74

	y_top = img_size[1] * top_off
	y_bot = img_size[1] * bot_off

	x_left_top  = img_size[0] * ( 0.5 - center_top_off) # width/2 - width*off
	x_right_top = img_size[0] * ( 0.5 + center_top_off) # width/2 + width*off

	x_left_bot  = img_size[0] * ( 0.5 - center_bot_off) # width/2 - width*off
	x_right_bot = img_size[0] * ( 0.5 + center_bot_off) # width/2 + width*off

	x_right_bot += x_right_bot_corr
	x_right_top += x_right_top_corr


	#test code
	#test on test_6 image where the lines are straight , aims was to get the points on the lane line
	src = np.array( [ [x_left_top,y_top] , [x_right_top,y_top] , [x_right_bot,y_bot] , [x_left_bot,y_bot] ] , np.int32).reshape((-1,1,2))
	tmp = cv2.polylines(img_undist , src, isClosed = True , thickness = 10 ,color = (0,200,100))
	cv2.imwrite(img_file.replace("test_images/test" , "output_images/test_pol_") , tmp)

	src = np.float32( [ [x_left_top,y_top] , [x_right_top,y_top] , [x_right_bot,y_bot] , [x_left_bot,y_bot] ] ) 

	dst_x_off = 250
	dst_y_off = 100

	dst = np.float32([ [ dst_x_off , 1.2 * dst_y_off ] , [ img_size[0] - dst_x_off, 1.2 * dst_y_off ] , [ img_size[0] - dst_x_off , img_size[1] - dst_y_off ] , [ dst_x_off , img_size[1] - dst_y_off ]  ])

	pres_mtx = cv2.getPerspectiveTransform(src , dst)
	warped = cv2.warpPerspective(combined , pres_mtx , (1280 , 720) , flags=cv2.INTER_LINEAR)


	cv2.imwrite( img_file.replace("test_images/test" , "output_images/test_binary_") , combined)
	cv2.imwrite( img_file.replace("test_images/test" , "output_images/test_warped_") , warped)

