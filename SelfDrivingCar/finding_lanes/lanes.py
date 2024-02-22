#Imported OpenCV and Numpy Libraries.
import cv2 #For image processing
import numpy as np #np is called as alias and for Numerical operations


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept)) 

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


#Function to return edge detected Image
#Function Canny take an image called lane_image, and do Edge detection.
def canny(image):
    #Convereted to Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #Smoothened and recuded noise using 5*5 Kernel
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #Edge detecting using Canny detector.
    canny= cv2.Canny(blur,50,150)
    return canny

#This defines a function named display_lines that takes an image image and a set of lines lines as input. 
#It creates a blank image (line_image) of the same size as the input image. 
#If there are detected lines, it iterates over each line, extracts its coordinates, and draws the line on the blank image. 
#It then returns the image with lines drawn on it.

def display_lines(image, lines):
    #initializes an empty image (line_image) with the same dimensions and data type as the original image.
    line_image = np.zeros_like(image)
    #This checks if there are any lines detected in the image. 
    #If there are lines (lines is not None), the code proceeds to draw them; otherwise, it skips the drawing part.
    if lines is not None:
        #This extracts the coordinates of the detected line. 
        #The reshape(4) function reshapes the line from a 2D array to a 1D array with 4 elements, representing the endpoints of the line.
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            #This draws a line on the line_image using the extracted coordinates (x1, y1) and (x2, y2). 
            #The color of the line is specified as (255,0,0), which is blue in OpenCV's BGR color format. 
            #The thickness of the line is set to 10.
            cv2.line(line_image, (x1,y1), (x2, y2), (255,0,0), 10)
    return line_image


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
    # Use bitwise_and to mask the image(Cropping)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


#Loaded image to a variable image
#image = cv2.imread('Image/test_image.jpg')

#Created a copy of image.
#lane_image = np.copy(image)

#Calling function Canny
#canny_image = canny(lane_image)

#Applying region_of_interest function to edge detedtec image(canny)
#cropped_image = region_of_interest(canny_image)

#Detects lines in croppped image using Hough Transform. Returns list of lines represented by their endpoints.
#lines = cv2.HoughLinesP(Image to be detected, resolution of parameter(row in pixels), resolution of parameter(0 in radians),
# threshold parameter, array to store detected line, min line length, max line gap)
#lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap=5)

#averaged_lines = average_slope_intercept(lane_image, lines)

#Calling function to draw the detected lines on original image and stored in line_image
#line_image= display_lines(lane_image, averaged_lines)

#Combine original_image & detected line image.
#combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

#Shows image in a window called result
#cv2.imshow('result', combo_image)

#Displays untill a key is pressed
#cv2.waitKey(0)

# Added video
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _ ,frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image= display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()