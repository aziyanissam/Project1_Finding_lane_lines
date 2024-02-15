#Imported OpenCV and Numpy Libraries.
import cv2 #For image processing
import numpy as np #np is called as alias and for Numerical operations


#Function to return edge detected Image
def canny(lane_image):
    #Convereted to Grayscale image
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    #Smoothened and recuded noise using 5*5 Kernel
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #Edge detecting using Canny detector.
    canny= cv2.Canny(blur,50,150)
    return canny

    #Function Canny take an image called lane_image, and do Edge detection.

#Function to return enclosed region
def region_of_interest(image):
    #Calculate height of input image.
    height =  image.shape[0]
    #DEfine vertices of polygon.
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    #This creates an array filled with zeros(full black), matching dimension of input image.
    mask = np.zeros_like(image)
    #Fills defined polygon in the mask array with 255(Triangle polygon to white)
    cv2.fillPoly(mask,polygons,255)
    return mask


#Loaded image to a variable image
image = cv2.imread('Image/test_image.jpg')

#Created a copy of image.
lane_image = np.copy(image)

#Calling function Canny
canny = canny(lane_image)

#Shows image in a window called result
cv2.imshow('result', region_of_interest(canny))

#Displays untill a key is pressed
cv2.waitKey(0)

#Imported OpenCV and Numpy Libraries.
#import cv2
#import numpy as np

#Loaded image to a variable image
#image = cv2.imread('Image/test_image.jpg')

#Created a copy of image.
#lane_image = np.copy(image)

#Convereted to Grayscale image
#gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

#Smoothened and recuded noise using 5*5 Kernel
#blur = cv2.GaussianBlur(gray,(5,5),0)

#Edge detecting using Canny detector.
#canny= cv2.Canny(blur,50,150)

#Shows image in a window called result
#cv2.imshow('result', canny)

#Displays untill a key is pressed
#cv2.waitKey(0)
