import cv2
import numpy as np
from Light import Light
import math
#from utils import *
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
useyolo =0
if useyolo ==1:
    from yolo_pipeline import *
#from lane import *
#import skvideo.io
from cv2 import __version__
print(__version__)

def ConverCars1toCars(Cars1,cars):
    retArray =[]
    cars =list(cars)
    for i in range(0,len(Cars1)):
        now =[]
        thisCar = Cars1[i]
        now.append(thisCar[0][0])
        now.append(thisCar[0][1])
        now.append(thisCar[1][0])
        now.append(thisCar[1][1])
        cars.append((thisCar[0][0],thisCar[0][1],thisCar[1][0],thisCar[1][1]))
    return tuple(cars)

def pipeline_yolo(img):
    img = cv2.resize(img, (1280, 720))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    #print(img.shape)

    #img_undist, img_lane_augmented, lane_info = lane_process(img)
    window_list = vehicle_detection_yolo1(img)#vehicle_detection_yolo(img_undist, img_lane_augmented, lane_info)

    return window_list

car_cascade = cv2.CascadeClassifier('cars3.xml')
cap = cv2.VideoCapture('part3_28.mp4')
#cap = skvideo.io.VideoCapture('DayRedLight1.mp4')
width = int(cap.get(3))
height = int(cap.get(4))
FrameRate=cap.get(5)
#length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
filename='part3_28out1.avi'
#fourcc = cv2.cv.CV_FOURCC(*'XVID')#
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter(filename, fourcc, FrameRate, (1280,720))
count = 0
font = cv2.FONT_HERSHEY_SIMPLEX
color = np.random.randint(0,255,(5000,3))


def UpdateLightList(LighList,cx,cy):
    distance = float(1000)
    distance_now = float(1000)
    lightnum = - 1
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
def TrackLights(contours,LighList):
    for i in range(0, len(contours)):
        cnt = contours[i]
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
        LighList = UpdateLightList(LighList,cx,cy)
        return LighList
def DetectInBox(x1,y1,x2,y2,cars):
    for (x, y, w, h) in cars:
        if w * h > 10000:
            if x1>x and x1<x+w and x2>x and x2<x+w and y1>y and y1<y+h and y2>y and y2<y+h:
                return True
    return False


def DrawLightCenters(LighList,img_copy1):
    if LighList is not None:
        for i in range(0,len(LighList)):
            cv2.circle(img_copy1, (LighList[i].Realworld_PosX[-1], LighList[i].Realworld_PosY[-1]), 5, color[LighList[i].Identity].tolist(), -1)
            cv2.putText(img_copy1, str(LighList[i].Identity), (LighList[i].Realworld_PosX[-1], LighList[i].Realworld_PosY[-1]), font, 1, color[LighList[i].Identity].tolist(), 2, cv2.CV_AA)
    return img_copy1

def FindPairs(contours,distmin,distmax,slopemin,slopemax,img_copy1,cars,slopefile,startM,endM,count ):
    circle_centers_X = []
    circle_centers_Y = []
    counterlist =[]
    for i in range(0,len(contours)):
        counterlist.append(-1)
    for i in range(0, len(contours)):
        cnt = contours[i]
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
    for i in range(0,len(circle_centers_X)):
        for j in range(i+1,len(circle_centers_X)):
            dist = math.hypot(circle_centers_X[j] - circle_centers_X[i], circle_centers_Y[j] - circle_centers_Y[i])
            #print "dist is "+str(dist)
            if dist > distmin and dist <distmax:
                if abs(circle_centers_X[j] - circle_centers_X[i])>=0.01:
                    slope = (circle_centers_Y[j] - circle_centers_Y[i])/(circle_centers_X[j] - circle_centers_X[i])
                    #print slope
                    if slope >slopemin and slope<slopemax:
                        counterlist[i]=j
                        if  (DetectInBox(circle_centers_X[i], circle_centers_Y[i], circle_centers_X[j], circle_centers_Y[j], cars)):
                            cv2.line(img_copy1, (int(circle_centers_X[i]), int(circle_centers_Y[i])), (int(circle_centers_X[j]), int(circle_centers_Y[j])), [255,255,0], 2)
                            slopefile.write(str(startM )+','+str(endM )+','+str(count )+','+str(circle_centers_X[i]) + ',' + str(circle_centers_Y[i]) + ',' + str(circle_centers_X[j]) + ',' + str(circle_centers_Y[j])+ '\n')
                            break
    return [circle_centers_X, circle_centers_Y, counterlist,img_copy1]

startM = 200
endM = 5000#length
print(endM)
distmin = 20
distmax = 500
slopemin = -0.3
slopemax = 0.3
datadump = 'Horizantal_Harr_part3_28' + str(startM) +str(endM) + '.csv'
slopefile = open(datadump, 'w')
LighList = []
while(count<endM):
    if count%500==0:
        LighList = []
    if count <=startM:
        ret, old_frame = cap.read()
        #print(ret)
        old_frame = cv2.resize(old_frame, (1280, 720))
        count = count + 1
    else:
        ret, old_frame = cap.read()
        old_frame = cv2.resize(old_frame, (1280, 720))
        count = count + 1
        img_copy1 = old_frame.copy()
        img_copy2 = old_frame.copy()
        old_gray = cv2.cvtColor(img_copy2, cv2.COLOR_BGR2GRAY)
        print(count)
        hsv = cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)
        #lower_red = np.array([160, 100, 100])
        lower_red = np.array([160, 100, 100])
        upper_red = np.array([179, 255, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_red, upper_red)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img_copy2,img_copy2, mask= mask)
        graymasked = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(graymasked, (11, 11), 0)
        #ret, thresh = cv2.threshold(blurred, 230, 255, 0)
        im2, contours, hierarchy = cv2.findContours(blurred , 1, 2)

        #LighList = TrackLights(contours, LighList)
        #DrawLightCenters(LighList, img_copy1)
        #print len(contours)
        
        cv2.drawContours(img_copy1, contours, -1, (0, 255, 0), 3)
        cars = car_cascade.detectMultiScale(old_gray, 1.3, 3)
        '''
        cars1 = pipeline_yolo(img_copy2)
        if len(cars1) >0:
            cars = ConverCars1toCars(cars1,cars)
            #print(cars1List)
            #cars = cars + cars1List
	    '''
        for (x, y, w, h) in cars:
            if w*h >10000:
                cv2.rectangle(img_copy1, (x, y), (x + w, y + h), (255, 0, 0), 2)

        [circle_centers_X, circle_centers_Y, counterlist,img_copy1] = FindPairs(contours, distmin, distmax, slopemin, slopemax,img_copy1,cars,slopefile,startM,endM,count)
        out.write(img_copy1)
    #out.write(res)
