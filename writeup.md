**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image7]: ./camera_cal/calibration3.jpg "Distored image"
[image8]: ./camera_cal_output/calibration3.jpg "Undisoted image"
[image9]: ./output_images/binary_test3.jpg "Binary image"
[image10]: ./output_images/binary_test3.jpg "straight lines"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

a part of the submitted git repo.

### Camera Calibration

The code for this step is contained in the file code/01-camera-cal.py

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result as in "camera_cal_output" there was an issue that not all input images returned true when calucating the distortion matrix because of the unmatching number of corners: 

![alt text][image7] ![alt text][image8]

### Pipeline (single images)

#### 1. correct distoartion of images 
in the code "code/process_image.py" (line 277-280) the pickle where the distortion matrix was calucated is opened and loaded 
then applied to every image (line 287) imported by the glob object in a a for loop (line 285)

#### 2. apply color transforms and  gradients to create a thresholded binary image.

the main function to proccess an image "handle_undistored_image" is called (line 289)
in this function a binary image is obtained by applying sobel opertor in X direction (line 65)
and also in Y direction (line 69)
color threshold (line 72) which is done by both the V channel in HSV and the L channel in HLS and the output is the anded result of both filter.

the final binary image is obtained when the gradient in X and Y direction (both) are in the detection threshold -gradient condtion- or if it was deteced by the color masking 

the final output binary is similar to this : 
![alt text][image9] 


#### 3. perspective transform.

in order to get the prespective transform the straight lane lines images were used 
the startgey was to place (by trial and error) four corners on the lines forimg a Trapezoid and then map those to a rectangle and get the prespective matrix of that mapping and apply it to other images "code/process_image.py" (lines 84-125)

to validate this iamge was printed 
![alt text][image10] TOBEDONE

#### 4.identifing lane-line pixels and fit the positions with a polynomial.

after the image was warped it was assumed that a big part of the unnedded infortion was droped due to the warpping things like the horizon lines , other cars and signs which lead to focus only on the potion of the road that contains the lane line like this image
![alt text][image11] TOBEDONE

how this was done was by taking a histogram of white points in the the lower half of the image ( the array indexing is only from the width//2) (line 133)


then the two histogram peaks for right center and left center are calcauted (line 137-138)

after the center is obtained sliding window hapens for 9 window by dividing the image to 9 sections and doing the searchs in a for loop for each sections (line 159)

for every window if the number of point that are marked is more than a certain threshold "shift_thres" (line 179-182) the center is shifted to the center of those points 

then all the found point for all the windows are appended in the same list (line 185-186)

after this for loop we now have two lists that have all the points that were in the slidings search windows that were used 
those are passed to the numpy function fit to find the polyline that best fits those with the assumpation that a second degree line will fit them the best (line 196-197)


#### 5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.

in the same file "code/process_image.py" (lines 261-262) using the equation provided to calucated the radius of curvuture and after making an assumption for the density for meter per pixel the esimated raduis of the road was calucated once for the fight curve and once for the left one and the output was the average of both of the obtained numbers 

camera center was calcauted to be the center btween the right and left lane as seen in the code (line 265-266)

#### 6. final output.

the fnal return image was the input image with overlayed , warped image of the lanes in binary in the upper right corner , the inforation for the camera offest the center and the estimated road curvature on the top and an overlay to mark the detected lanes in green with mostly invisble red lines to mark the lanes line themsevles 

also there was an output of the mask generated before and after the warping it back with the inverse transfom

![alt text][image12] TOBEDONE

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
