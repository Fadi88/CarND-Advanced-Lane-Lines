import cv2
import numpy as np
import glob
import pickle

#define number of corners in X and Y directions
nx = 9
ny = 6

#create a list for standrad object points in 3D and set the Z to zeros to append every time an image is calibrated 
objp = np.zeros(( nx*ny , 3) , np.float32)
objp[:,:2] = np.mgrid[0:nx , 0:ny].T.reshape(-1 , 2)

#create object points and image points empty lists to be filled in a loop over the calibrations images
objpoints = [] #3d points evrey time a return is true will be filled with the list objp
imgpoints = [] #2d points found by opencv

#create and image pickle to loop on every calibration image
img_pickle = glob.glob('../camera_cal/calibration*.jpg')

for img_file in img_pickle:
	img = cv2.imread(img_file , flags = cv2.IMREAD_GRAYSCALE)

	#find corners using opencv
	corners_found_b , corners_list = cv2.findChessboardCorners(img , (nx , ny) , None)

	if corners_found_b is True:
		print(img_file , ' is being processed')
		objpoints.append(objp)
		imgpoints.append(corners_list)

		#draw found corners on the image and save it in the output folder
		cv2.drawChessboardCorners(img , (nx ,ny) , corners_list , True)
		cv2.imwrite(img_file.replace("camera_cal" , "camera_cal_output") , img)
	else:
		print(img_file, 'failed to detect corners !!')

img_size = (img.shape[1] , img.shape[0])
_ , camera_matrix , distoration_matrix , _ , _ = cv2.calibrateCamera( objpoints , imgpoints , img_size , None , None )

dump_dict = {}
dump_dict['mtx'] = camera_matrix
dump_dict['dist'] = distoration_matrix
pickle.dump(dump_dict , open('../camera_cal_output/cal-pickle.p' , 'wb'))


