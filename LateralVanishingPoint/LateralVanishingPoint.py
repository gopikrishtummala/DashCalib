'''
------------------------------------------------------------------------------
--------------Part of DashCalib project at The Ohio State University----------
--------------File Name: LateralVanishingPoint.Py -----------------------------
Takes in a video and identifies the taillights to find the average slope.
For debugging and visualizations it taillights to the output video..

Usage -  LateralVanishingPoint.py

--------------Authors: Gopi Krishna Tummala, Prasun Sinha and, Tanmoy Das-----
--------------Contact email: tummala.10@osu.edu, prasun@cse.ohio-state.edu
--------------For copyrights please contact the authors----------------------

------------------------------------------------------------------------------
ToDo:
------------------------------------------------------------------------------
'''
import cv2
import numpy as np
from Light import Light
import math
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
#from utils import *
#from lane import *
#import skvideo.io

#.....Printing the OpenCV version, Currently the code works well with OpenCV 3.3.0
from cv2 import __version__
print(__version__)

#.......If you are using the YOLO pipeline set the useyolo flag to "0"
useyolo =1
#.....Flag to use YOLO pipeline
if useyolo ==0:
    from yolo_pipeline import *


#.....Method that coverts the bounding-boxes (cars) returned by the YOLO pipeline to the same format as bounding boxes returned by HAAR based classifier.
def ConverCars1toCars(Cars1,cars):
    #..Convert the cars from "tuple" to list to append the "Cars1", then we will convert back to tuple and return.
    cars =list(cars)

    #.. For each car in Cars1, get the x,y,w,h
    for i in range(0,len(Cars1)):
        now =[]
        #...Empty array for appending the atributes of car in Cars1
        thisCar = Cars1[i]
        now.append(thisCar[0][0])
        now.append(thisCar[0][1])
        now.append(thisCar[1][0])
        now.append(thisCar[1][1])
        cars.append((thisCar[0][0],thisCar[0][1],thisCar[1][0],thisCar[1][1]))
    return tuple(cars)

#...Method to run the YOLO pipe line on an image.
def pipeline_yolo(img):
    #........Resizing the image as the YOLO pipeline
    img = cv2.resize(img, (1280, 720))
    #........Converting the image to grey scale.
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    #.......If there is distortion to the camera, remove the distortion.
    #img_undist, img_lane_augmented, lane_info = lane_process(img)
    window_list = vehicle_detection_yolo1(img)#vehicle_detection_yolo(img_undist, img_lane_augmented, lane_info)
    return window_list

#.....Method to update the LightList based on the current light (cx,cy)...#
def UpdateLightList(LighList,cx,cy):
    distance = float(1000)
    #.. If lightList has no elements create the light object and add to the lightList
    if LighList is None:
        LighList =[]
        Createlight = Light.Light.Light()
        Createlight.Identity =  1
        Createlight.UpdateLightPos(cx,cy)
        LighList.append(Createlight)
    else:
        fa =len(LighList)
        lightnum = -1
        for j in range(1, fa):
            CurSearch = LighList[j]
            x = CurSearch.Realworld_PosX[-1]
            y = CurSearch.Realworld_PosY[-1]
            x1 = cx
            y1 = cy
            distance_now = math.hypot(x1 - x, y1 - y)
            if (distance_now < distance and distance_now < 30):
                distance = float(distance_now)
                lightnum = j
        if (lightnum == - 1):
            Createlight = Light.Light.Light()
            Createlight.Identity = fa + 1
            Createlight.UpdateLightPos(cx,cy)
            LighList.append(Createlight)
        else:
            LighList[lightnum].UpdateLightPos(cx,cy)
    return LighList

#.....Method for tracking the lights across the frames based on the derived contours...#
def TrackLights(contours,LighList):

    #.....For each contour we obseved.......#
    for i in range(0, len(contours)):
        cnt = contours[i]

        #.. Deriving the moments of the contour
        M = cv2.moments(cnt)
        #...Deriving the area of the contour
        Area = cv2.contourArea(contours[i])

        #...Finding the center of the contour.......#
        if M['m00'] == 0:
            cx = 0
            cx_float = 0.0
            cy = 0
            cy_float = 0.0
        else:
            cx_float = M['m10'] / M['m00']
            cy_float = M['m01'] / M['m00']
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        # Updating the LightList based on the current contour center
        LighList = UpdateLightList(LighList,cx,cy)
        return LighList

#......Method to detect if the two lights belong to a car.
def DetectInBox(x1,y1,x2,y2,cars):
    # For each car belonging to the cars-list (cars)
    for (x, y, w, h) in cars:
        #.If the size of the car is greater than minimum threshold.
        if w * h > 10000:
            #Checking if the both the points belonging to the current car
            if x1>x and x1<x+w and x2>x and x2<x+w and y1>y and y1<y+h and y2>y and y2<y+h:
                return True
    return False


def DrawLightCenters(LighList,img_copy1):
    if LighList is not None:
        for i in range(0,len(LighList)):
            cv2.circle(img_copy1, (LighList[i].Realworld_PosX[-1], LighList[i].Realworld_PosY[-1]), 5, color[LighList[i].Identity].tolist(), -1)
            cv2.putText(img_copy1, str(LighList[i].Identity), (LighList[i].Realworld_PosX[-1], LighList[i].Realworld_PosY[-1]), font, 1, color[LighList[i].Identity].tolist(), 2, cv2.CV_AA)
    return img_copy1

#... Method to identify the red-light pairs (taillights of the same vehicle)....#
def FindPairs(contours,distmin,distmax,slopemin,slopemax,img_copy1,cars,slopefile,startM,endM,count ):

    #......List for storing the centers of the taillights.
    circle_centers_X = []
    circle_centers_Y = []
    counterlist =[]
    for i in range(0,len(contours)):
        counterlist.append(-1)

    #....Update the centers list by deriving the centers of the contours.
    for i in range(0, len(contours)):
        cnt = contours[i]
        # .. Deriving the moments of the contour
        M = cv2.moments(cnt)
        Area = cv2.contourArea(contours[i])
        if M['m00'] == 0:
            cx = 0
            cx_float = 0.0
            cy = 0
            cy_float = 0.0
        else:
            cx_float = M['m10'] / M['m00']
            cy_float = M['m01'] / M['m00']
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            circle_centers_X.append(cx_float)
            circle_centers_Y.append(cy_float)

    #....Analyze the contours lists to identify the taillight pairs.
    for i in range(0,len(circle_centers_X)):
        for j in range(i+1,len(circle_centers_X)):

            #.......Derive the distance between the two contour centers .........#
            dist = math.hypot(circle_centers_X[j] - circle_centers_X[i], circle_centers_Y[j] - circle_centers_Y[i])

            #.......Threshold for the line joining the taillights...............#
            if dist > distmin and dist <distmax:

                #...If the circles are separated along the X-axis (Just one more threshold, can be commented out
                if abs(circle_centers_X[j] - circle_centers_X[i])>=0.01:

                    #...Deriving the slope of the line joining the taillights.
                    slope = (circle_centers_Y[j] - circle_centers_Y[i])/(circle_centers_X[j] - circle_centers_X[i])
                    #print slope

                    #..... Threshold for the slope of the line joining the taillights.
                    if slope >slopemin and slope<slopemax:

                        #....See if the two lights belonging a bounding box. If the two lights belonging to some box they are grouped
                        # a taillights belonging to the same vehicle.
                        #....Note: There can be multiple lines joining to the vehicle.. Need to design algorithms to improve the selection
                        if  (DetectInBox(circle_centers_X[i], circle_centers_Y[i], circle_centers_X[j], circle_centers_Y[j], cars)):
                            #.... Update the grouping matrix.
                            counterlist[i] = j
                            #.....Draw the line on output video depicting these grouping
                            cv2.line(img_copy1, (int(circle_centers_X[i]), int(circle_centers_Y[i])), (int(circle_centers_X[j]), int(circle_centers_Y[j])), [255,255,0], 2)
                            #.....Update the log file for further analysis.
                            slopefile.write(str(startM )+','+str(endM )+','+str(count )+','+str(circle_centers_X[i]) + ',' + str(circle_centers_Y[i]) + ',' + str(circle_centers_X[j]) + ',' + str(circle_centers_Y[j])+ '\n')
                            break
    #.....Return the counter center list, their grouping, and the drawn image.
    return [circle_centers_X, circle_centers_Y, counterlist,img_copy1]

car_cascade = cv2.CascadeClassifier('cars3.xml')
#....... Define the haar cascade classifier for vehicles..........#

cap = cv2.VideoCapture('../input.mp4')
#cap = skvideo.io.VideoCapture('DayRedLight1.mp4')
#....... Define the input video for processing..........#

width = int(cap.get(3))
height = int(cap.get(4))
FrameRate=cap.get(5)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#....... Derive the attributes of video reader object..........#

filename='../outputVP2.avi'
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
#fourcc = cv2.cv.CV_FOURCC(*'XVID')# Use this line for some version of opencv. For these versions there will be error in the above line
out = cv2.VideoWriter(filename, fourcc, FrameRate, (1280,720))
#....... Define the output video for drawing which will help in debugging and analysis..........#


font = cv2.FONT_HERSHEY_SIMPLEX
color = np.random.randint(0,255,(5000,3))
#....... Defining the colors and font for drawing on the output video..........#

startM = 3000
count = startM
endM = 5000
print(str(startM)+','+str(endM)+'\n')
#.......Define the start and end frames to be processed
cap.set(1,startM )
#... Set the cap object to the startM frame number

#.... Define the threshold properties for lines joining the taillights.
distmin = 20
#..The minimum distance between the two taillights
distmax = 500
#..The maximum distance between the two taillights
slopemin = -0.3
#..The minimum slope of the lines joining the taillights
slopemax = 0.3
#..The maximum slope of the lines joining the taillights

#.....Defining the lower RED and upper RED colors for identifying the taillights.
lower_red = np.array([160, 100, 100])
upper_red = np.array([179, 255, 255])

#......Log file for writing the lines joining the taillights.
datadump = './Horizantal_DNN' + str(startM) +str(endM) + '.csv'
slopefile = open(datadump, 'w')

LighList = []
while(count<endM):
        ret, old_frame = cap.read()
        #.......Reading the frame from the video-reader object
        count = count + 1
        print(count)
        #.......Incrementing the count to keep track of which frame is being processed.
        old_frame = cv2.resize(old_frame, (1280, 720))
        #.......Resizing the frames to reduce the computation.

        #......Creating the copies of the frame for trying different algorithms.
        img_copy1 = old_frame.copy()
        img_copy2 = old_frame.copy()

        old_gray = cv2.cvtColor(img_copy2, cv2.COLOR_BGR2GRAY)
        #.....Converting the frame to gray scale.........
        hsv = cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)
        #.....Converting the frame to hsv color space.........

        # Threshold the HSV image to get only red colored blobs
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img_copy2,img_copy2, mask= mask)

        # Converting the residual image to gray scale..
        graymasked = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        # Smoothing the image using the Gaussian blurr..
        blurred = cv2.GaussianBlur(graymasked, (11, 11), 0)

        # Finding contours using the blurred image......
        im2, contours, hierarchy = cv2.findContours(blurred , 1, 2)

        #.. Drawing the contours on the output video file.
        cv2.drawContours(img_copy1, contours, -1, (0, 255, 0), 3)

        #.. Drawing the vehicles in the current frame...
        cars = car_cascade.detectMultiScale(old_gray, 1.3, 3)

        #...If using the YOLO vehicle Detection DNN.....#
        if useyolo == 0:
            # ...Identify the vehicle bounding boxes by running YOLO pipeline.....#
            cars1 = pipeline_yolo(img_copy2)
            # ...if YOLO pipeline found vehicles, convert the bounding boxes to the same format as the bounding boxes returned by Haar based vehicle detector.....#
            if len(cars1) > 0:
                cars = ConverCars1toCars(cars1, cars)

        #......Draw the rectangles to the output video for debugging purposes....#
        for (x, y, w, h) in cars:
                #.. If the bounding box has significant size..
                if w * h > 10000:
                    cv2.rectangle(img_copy1, (x, y), (x + w, y + h), (255, 0, 0), 2)


        [circle_centers_X, circle_centers_Y, counterlist,img_copy1] = FindPairs(contours, distmin, distmax, slopemin, slopemax,img_copy1,cars,slopefile,startM,endM,count)
        out.write(img_copy1)
        #LighList = TrackLights(contours, LighList)
        #DrawLightCenters(LighList, img_copy1)
        #print len(contours)
    #out.write(res)
