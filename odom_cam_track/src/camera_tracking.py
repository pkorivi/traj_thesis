
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    #return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def hsv(img):
    #return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    #cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def get_red_thresh_img(p_img):
    #Red_Thresholds HSV
	lower_red1 = np.array([0, 100, 100])
	upper_red1 = np.array([10, 255,255])
	lower_red2 = np.array([160,100,100])
	upper_red2 = np.array([179,255,255])
    """
    #in BGR
    lower_red1 = np.array([104, 27, 152])
    upper_red1 = np.array([205,255,255])
    """
    return cv2.inRange(p_img, lower_red1, upper_red1)

def get_green_thresh_img(p_img):
    lower_green = np.array([60,60,46])
	upper_green = np.array([97,255,255])
    """
    #in BGR
    lower_green = np.array([70,108,128])
    upper_green = np.array([163,255,255])
    """ #in HSV
	# Threshold the HSV image to get only single color portions
    return cv2.inRange(p_img, lower_green, upper_green)

def get_contours(thresh_img):
    _, contours, hierarchy = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_squares(contours):
    coordi = []
    #coordi = np.array([],dtype=object)
    #coordi = np.ndarray([],dtype=object)
    for x in range (len(contours)):
        contourarea = cv2.contourArea(contours[x]) #get area of contour
        if contourarea > 300: #Discard contours with a small area as this may just be noise
            #The below 2 functions help you to approximate the contour to a nearest polygon
            arclength = cv2.arcLength(contours[x], True)
            approxcontour = cv2.approxPolyDP(contours[x], 0.02 * arclength, True)
            #Check for Square
            if len(approxcontour) == 4:
                #Find the coordinates of the polygon with respect to he camera frame in pixels
                rect_cordi = cv2.minAreaRect(contours[x])
                obj_x = int(rect_cordi[0][0])
                obj_y = int(rect_cordi[0][1])
                #print(approxcontour)
                coordi.append((obj_x,obj_y))

                #coordi=np.concatenate(coordi,(obj_x,obj_y))
                #print ('Length ', len(approxcontour))
                    #approxcontour = approxcontour.reshape((4,2))
                #cv2.drawContours(cv_image,[approxcontour],0,(0,255,255),2)
                #LongestSide = Longest_Length(approxcontour)
                #Distance = (focal_leng*square_side_lenth)/LongestSide #focal length x Actual Border width / size of Border in pixels
            #Move to next Contour
            else :
                continue

            #Calculate Cordinates wrt to Camera, convert to Map
            #Coordinates and publish message for storing
            #319.5, 239.5 = image centre
            #obj_cam_x = ((obj_x - 319.5)*Distance)/focal_leng
            #obj_cam_y = ((obj_y - 239.5)*Distance)/focal_leng
    return coordi

def estimate_car_coordinates(g_sq, r_sq):
    #implement some mechanism to reduce error and estimate position of the car such that it starts at (0,0) and plots similar to
    return (0,0)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    hsv_img = hsv(image)
    g_thresh_img = get_green_thresh_img(image)
    g_contours = get_contours(g_thresh_img)
    g_squares = get_squares(g_contours) #coordinates of square

    r_thresh_img = get_red_thresh_img(image)
    r_contours = get_contours(r_thresh_img)
    r_squares = get_squares(r_contours) #coordinates of square

    car_cordi = estimate_car_coordinates(g_squares, r_squares)

    """
    implement mechanism to write to a file - the estimated coordinates
    """
    #"""
    print("g_squares :: ")
    print(g_squares)
    print("r_squares :: ")
    print(r_squares)
    #"""
    for pt in r_squares:
        #cv2circle(image, pt, 20, (255,0,0), thickness=1, lineType=8, shift=0)
        cv2.circle(image,pt, 25, (0,0,255), -1)
    for pt in g_squares:
        cv2.circle(image,pt, 25, (0,255,0), -1)
    #cv2.imshow("Image",r_thresh_img)
    #cv2.imshow("HSV", hsv)
    #cv2.waitKey()
    return image


"""
Main program
"""

"""
image = cv2.imread('img_vid/im3.jpg')
f_img = process_image(image)
#cv2.imshow("final", f_img)
cv2.imwrite("img_vid/fi.jpg",f_img)
#cv2.waitKey()

white_output = 'img_vid/vid1_0.mp4'
clip1 = VideoFileClip("img_vid/vid2.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
"""

def callback(data):
    global cv_image
    bridge = CvBridge()
    #Use the below code for Turtlebot compressed image
    #np_arr = np.fromstring(data.data, np.uint8)
    #cv_image = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
    #If subscribing to Drone use the below line
    print 'callback'
    """
    np_arr = np.fromstring(data.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    final_img = process_image(cv_image)
    track_pub.publish(final_img)
    """
"""
def main(args):
    rospy.init_node('hsv_picker', anonymous = False)
    ic = hsv_picker()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down hsv_picker Node')
        cv2.destroyAllWindows()
        video.release()

if __name__ == '__main__':
	main(sys.argv)
"""

def main(args):
    rospy.init_node('camera_tracker', anonymous=False)
    image_sub = rospy.Subscriber("/app/camera/rgb/image_raw/compressed",CompressedImage, callback)
    track_pub = rospy.Publisher("/track_img", Image, queue_size =1)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rospy.spin()
        rate.sleep()

if __name__ == '__main__':
    main(sys.argv)
