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
	

def handle_undistored_image(img):
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
	img_height = img.shape[0]
	img_width = img.shape[1]
	img_size = ( img_width, img_height)

	top_off = 0.64
	bot_off = 0.9
	center_top_off = 0.045
	center_bot_off = 0.251

	x_right_bot_corr = 36
	x_right_top_corr = 10

	y_top = img_size[1] * top_off
	y_bot = img_size[1] * bot_off

	x_left_top  = img_size[0] * ( 0.5 - center_top_off) # width/2 - width*off
	x_right_top = img_size[0] * ( 0.5 + center_top_off) # width/2 + width*off

	x_left_bot  = img_size[0] * ( 0.5 - center_bot_off) # width/2 - width*off
	x_right_bot = img_size[0] * ( 0.5 + center_bot_off) # width/2 + width*off

	x_right_bot += x_right_bot_corr
	x_right_top += x_right_top_corr

	'''
	#test code
	#test on straight line image where the lines are straight , aims was to get the points on the lane line
	src = np.array( [ [x_left_top,y_top] , [x_right_top,y_top] , [x_right_bot,y_bot] , [x_left_bot,y_bot] ] , np.int32).reshape((-1,1,2))
	tmp = cv2.polylines(img_undist , src, isClosed = True , thickness = 10 ,color = (0,200,100))
	cv2.imwrite(img_file.replace("test_images/" , "output_images/pol_") , tmp)
	'''
	src = np.float32( [ [x_left_top,y_top] , [x_right_top,y_top] , [x_right_bot,y_bot] , [x_left_bot,y_bot] ] ) 

	dst_x_off = 250
	dst_y_off = 50

	dst = np.float32([ [ dst_x_off , 1.2 * dst_y_off ] , [ img_size[0] - dst_x_off, 1.2 * dst_y_off ] , [ img_size[0] - dst_x_off , img_size[1] - dst_y_off ] , [ dst_x_off , img_size[1] - dst_y_off ]  ])

	pres_mtx     = cv2.getPerspectiveTransform(src , dst)
	inv_pres_mtx = cv2.getPerspectiveTransform(dst , src)
	
	warped = cv2.warpPerspective(combined , pres_mtx , img_size , flags=cv2.INTER_LINEAR)

	cv2.imwrite( img_file.replace("test_images/" , "output_images/binary_") , combined)
	cv2.imwrite( img_file.replace("test_images/" , "output_images/warped_") , warped)

	#get histogram 
	histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0) #size image width * 1


	#caluclate basis for the search 
	left_lane_center  = np.argmax(histogram[:img_width//2])
	right_lane_center = np.argmax(histogram[img_width//2:]) + img_width//2
	
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	n_windows = 9
	window_height = img_height //n_windows
	window_half_width = 100
	out_img = np.dstack((warped, warped, warped))

	left_center_temp   = left_lane_center
	right_center_temp = right_lane_center

	shift_thres = 50 #in pixel

	#store points coordinates for both right and left lane 
	left_lane_inds  = []
	right_lane_inds = []

	for window_idx in range(0 , n_windows):
		y_base = img_height - (window_idx * window_height)
		y_top  = y_base - window_height
		
		left_window  = 	warped[ y_top : y_base ,  left_center_temp  - window_half_width :  left_center_temp  + window_half_width]
		right_window = 	warped[ y_top : y_base , right_center_temp  - window_half_width : right_center_temp  + window_half_width]

		new_left_center  = np.argmax(np.sum( left_window[ left_window.shape[0]//2:,:], axis=0))
		new_right_center = np.argmax(np.sum(right_window[right_window.shape[0]//2:,:], axis=0))
		
		cv2.rectangle(out_img , ( left_center_temp  - window_half_width , y_base) , ( left_center_temp  + window_half_width , y_top) , (0 , 255 , 0) , 2 )
		cv2.rectangle(out_img , ( right_center_temp - window_half_width , y_base) , ( right_center_temp + window_half_width , y_top) , (0 , 255 , 0) , 2 )
		
		#get all non zero points from both right and left
		good_left_inds  = ((nonzeroy >= y_top) & (nonzeroy < y_base) & (nonzerox >=  left_center_temp  - window_half_width) & (nonzerox <  left_center_temp  + window_half_width)).nonzero()[0]
		good_right_inds = ((nonzeroy >= y_top) & (nonzeroy < y_base) & (nonzerox >= right_center_temp  - window_half_width) & (nonzerox < right_center_temp  + window_half_width)).nonzero()[0]

		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		if len(good_left_inds) > shift_thres:
			left_center_temp = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > shift_thres:
			right_center_temp = np.int(np.mean(nonzerox[good_right_inds]))

	# flatten the list of list into a single list TODO check if reshape can be used here
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)


	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	# Generate x and y values for plotting
	ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	plt.show()

	cv2.imwrite( img_file.replace("test_images/" , "output_images/binary_") , combined)
	cv2.imwrite( img_file.replace("test_images/" , "output_images/warped_") , warped)

	return out_img

dict_pickle = pickle.load(open( "../camera_cal_output/cal-pickle.p", "rb"))

mtx  = dict_pickle['mtx']
dist = dict_pickle['dist']

#read all images as a glob object
img_glob = glob.glob("../test_images/test*.jpg")

for img_file in img_glob:
	img = cv2.imread(img_file)
	img_undist = cv2.undistort(img , mtx , dist , None , None)

	cv2.imwrite( img_file.replace("test_images/" , "output_images/boxed_") , handle_undistored_image(img_undist))
	
	


