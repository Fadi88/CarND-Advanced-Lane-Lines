import cv2
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
from process_image import *
from moviepy.editor import VideoFileClip



def process_frame(frame):
	dict_pickle = pickle.load(open( "../camera_cal_output/cal-pickle.p", "rb"))

	mtx  = dict_pickle['mtx']
	dist = dict_pickle['dist']

	img_undist = cv2.undistort(frame , mtx , dist , None , None)

	return handle_undistored_image(img_undist)

output_video = '../output_tracked.mp4'
input_video  = '../project_video.mp4'

clip1 = VideoFileClip(input_video)
out_obj = clip1.fl_image(process_frame)
out_obj.write_videofile( output_video , audio = False)	
	


