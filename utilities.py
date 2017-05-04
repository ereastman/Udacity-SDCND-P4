# utility functions
import numpy as np
import cv2
import glob
import math
import matplotlib.pyplot as plt

def calculateDistortionVals(image_filenames, border):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for fname in image_filenames:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        border_img = cv2.copyMakeBorder(gray, top=border[0], bottom=border[1], left=border[2], right=border[3], borderType=cv2.BORDER_CONSTANT) 
    
        # Find the chessboard corners
        
        ret, corners = cv2.findChessboardCorners(border_img, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    img_size = (border_img.shape[1], border_img.shape[0])

    # Compute camera calibration matrix        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    distort_vals = {'matrix': mtx, 'dist': dist}
    return distort_vals

def to_binary(img):
    xgradient=(20, 100)
    s_thresh=(170, 255)
    img_copy = np.copy(img)
    binary_img = apply_thresholding(img_copy, xgradient, s_thresh, False) 

    binary_img_blur = blur(binary_img, ksize=(7, 7), s_x=0, s_y=0)
    binary_img_b_thresh = threshold(binary_img_blur, (180, 255), to_val=255)
    
    return binary_img_b_thresh

def loadImage(fname, greyscale=False):
    img = cv2.imread(fname)
    if(greyscale):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = (img.shape[1], img.shape[0])
    return [img, img_size]

def show_images_in_folder(img_operator, foldername='./', glob_param='*', draw_points=False, pts=[]):
    # Make a list of images
    image_filenames = glob.glob(foldername+glob_param)
    num_cols = 2
    num_rows = math.ceil(len(image_filenames)/num_cols)

    plt.rcParams['figure.figsize'] = (20.0, 20.0)

    f, axarr = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)
    f.subplots_adjust(hspace=0.1)

    for row in np.arange(0,num_rows):
        ind=0
        for fname in image_filenames[row*num_cols:np.min([len(image_filenames),row*num_cols+num_cols])]:
            img, _ = loadImage(fname, greyscale=False)
            if draw_points == True:
              img = drawPoints(img, pts)
            final_img = img_operator(img)
            axarr[row, ind].imshow(final_img, cmap='gray')
            axarr[row, ind].set_title(fname)
            ind+=1

def showImage(img, figure_num=0, cmap='gray'):
    f = plt.figure(figure_num)
    plt.imshow(img, cmap='gray')
    return figure_num+1
   
def drawPoints(img, points, radius=5, color=[0, 0, 255]):
    new_img = np.copy(img)
    for p in points:
        cv2.circle(new_img, tuple(p), radius, color, thickness=3)
    return new_img

def undist(img, matrix, dist):
    return cv2.undistort(img, matrix, dist, None)

def transform_perspective(img, distort_vals, M, show_images=False):
    img_size = (img.shape[1], img.shape[0])
    
    # Warp onto birds-eye-view
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    figure_num=0
    if(show_images):
        print("Transformed Test Image")
        figure_num=showImage(warped, figure_num, cmap='gray')
    
    return warped

def threshold(img, thresh, to_val=1):
    # Apply thresholding
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) & (img <= thresh[1])] = to_val
    
    return binary
    
def abs_sobel_thresh(img, orientation='x', thresh=(0, 255)):
    # depending on input orientation, apply sobel op in that direction
    if orientation == 'x':
        derivative = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    elif orientation == 'y':
        derivative = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    
    # Take abs bc we dont care about direction
    abs_derivative = np.absolute(derivative)
    # Scale to 8-bit (0 - 255)
    scaled_sobel = np.uint8(255 * abs_derivative / np.max(abs_derivative))
    #print(scaled_sobel.shape) - 720, 1080
    # Apply thresholding
    #grad_binary = np.zeros_like(scaled_sobel)
    #grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    grad_binary = threshold(scaled_sobel, thresh)
    #print(grad_binary) - 1.023..
    
    # 6) Return this mask as your binary_output image
    return grad_binary

def saturation_thresh(color_img, thresh):
    # Convert to HLS colour space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(color_img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    s_binary = threshold(s_channel, thresh)
    return s_binary

def apply_thresholding(img, xgrad_thresh=(20,100), s_thresh=(170,255), show_images=False):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Threshold x-Gradient
    sxbinary = abs_sobel_thresh(gray, orientation='x', thresh=xgrad_thresh)
    
    # Threshold Saturation channel
    s_binary = saturation_thresh(img, s_thresh)
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 255
    #combined_binary = sxbinary*s_binary
    
    if(show_images):
        figure_num=0
        #print("Original Image")
        #figure_num=showImage(img, figure_num, cmap='gray')
        #print("Thresholded Sobel")
        #figure_num=showImage(sxbinary, figure_num, cmap='gray')
        #print("Thresholded Color")
        #igure_num=showImage(s_binary, figure_num, cmap='gray')
        print("Combined")
        figure_num=showImage(combined_binary, figure_num, cmap='gray')
        
    return combined_binary

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def blur(img, ksize, s_x, s_y):
    return cv2.GaussianBlur(img, ksize, sigmaX=s_x, sigmaY=s_y)

def fit_poly_1(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[round(binary_warped.shape[0]/5):,:], axis=0)
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 12
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 90
    # Set minimum number of pixels found to recenter window
    minpix = 0
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        

    # Concatenate the arrays of indices
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
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # Show Image
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    return [left_fit, right_fit, out_img]

def find_window_centroids(warped, window_width, window_height, margin):
    
    l_centroids = [] # Store the (left,right) window centroid positions per level
    r_centroids = []
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    height=warped.shape[0]
    width=warped.shape[1]
    conv_width=width+window_width-1
    l_confidence = 0
    r_confidence = 0
    l_center=-1
    r_center=-1
    # Go through each layer looking for max pixel locations
    for level in range(0,(int)(height/window_height)):
        # convolve the window into the vertical slice of the image
        yval = int(height-(level+0.5)*window_height)
        image_layer = np.sum(warped[int(height-(level+1)*window_height):int(height-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        #print(conv_signal)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = int(window_width/2)
        if(l_center == -1):
            l_min_index = 0
            l_max_index = int(conv_width/2)
            
        else:
            l_min_index = int(max(l_center-margin,0))
            l_max_index = int(min(l_center+margin,conv_width))
        l_ind = np.argpartition(conv_signal[l_min_index:l_max_index], -5)[-5:]
        #print("Left:  ", conv_signal[l_ind+l_min_index])
        #print(np.argsort(conv_signal[l_ind]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index
        # Find the best right centroid by using past right center as a reference
        if r_center == -1:
            r_min_index = int(conv_width/2)
            r_max_index = conv_width-1
        else:
            r_min_index = int(max(r_center-margin,0))
            r_max_index = int(min(r_center+margin,conv_width))
        r_ind = np.argpartition(conv_signal[r_min_index:r_max_index], -5)[-5:]
        #print("Right: ", conv_signal[r_ind+r_min_index], '\n')
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index
        # Add what we found for that layer
        if(conv_signal[l_center] > 100):
            l_centroids.append([yval, l_center-offset])
        else:
            l_center = -1
        if(conv_signal[r_center] > 100):
            r_centroids.append([yval, r_center-offset])
        else: 
            r_center = -1

    return np.array(l_centroids), np.array(r_centroids)

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def fit_poly_2(warped, window_height):
    # window settings
    window_width = 50 
    margin = 50 # How much to slide left and right for searching
    l_cent, r_cent = find_window_centroids(warped, window_width, window_height, margin)

    return l_cent, r_cent

def addLaneToOriginal(orig, blank_w_lane, Minv):
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(blank_w_lane, Minv, (blank_w_lane.shape[1], blank_w_lane.shape[0])) 
    # Convert to three channel
    color_warp = cv2.cvtColor(newwarp, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(orig, 1, color_warp, 0.3, 0)

def laneOverlay(img, l_window_centroids, r_window_centroids):
    left_fit_coes = np.polyfit(l_window_centroids[:, 0], l_window_centroids[:,1], 2)
    right_fit_coes = np.polyfit(r_window_centroids[:, 0], r_window_centroids[:,1], 2)
    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(start=0, stop=img.shape[0]-1, num=img.shape[0] )

    left_fitx = np.multiply(left_fit_coes[0],np.square(ploty)) + np.multiply(left_fit_coes[1],ploty) + left_fit_coes[2]
    right_fitx = np.multiply(right_fit_coes[0],np.square(ploty)) + np.multiply(right_fit_coes[1],ploty) + right_fit_coes[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    img_c = np.zeros_like(img)
    cv2.fillPoly(img_c, np.int_([pts]), (128))
    return img_c
