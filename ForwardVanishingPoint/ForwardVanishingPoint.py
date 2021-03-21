'''
------------------------------------------------------------------------------
--------------Part of DashCalib project at The Ohio State University----------
--------------File Name: ForwardVanishingPoint.Py-----------------------------
Takes in a vide and computes the forward vanishing point. For debugging and visualizations it draws the feature point tracks on the output video..

--------------Authors: Gopi Krishna Tummala, Prasun Sinha and, Tanmoy Das-----
--------------Contact email: tummala.10@osu.edu, prasun@cse.ohio-state.edu
--------------For copyrights please contact the authors----------------------

------------------------------------------------------------------------------
ToDo:
------------------------------------------------------------------------------
'''
import numpy as np
from scipy import stats
import cv2
import math
import os.path, time
from cv2 import __version__
import Feature
import VP
from optparse import OptionParser
'''
f_x=(2510.23 / 3264) * 1920
f_y=(2459.1 / 1836) * 1080
c_x=(1626.09 / 3264) * 1920
c_y=(977.91 / 1836) * 1080
'''
#............Setting up intrinsic parameters....................#
fx=(2892.2/ 3264) * 1920
fy=(2758.0 / 1836) * 1080
cx=(1661.1 / 3264) * 1920
cy=(1228.1 / 1836) * 1080
#...............................................................#

#............Method to dump feature points slope to the logfile.#
def WriteLines(FeatureVector,slopefile):
    for i in range(0, len(FeatureVector)):
        if FeatureVector[i].Mval != 0 and FeatureVector[i].Cval != 0:
            slopefile.write(str(FeatureVector[i].Mval) + ',' + str(FeatureVector[i].Cval) + '\n')

#............Method to update the current feature positions to the feature vector....#			
def UpdateFeatures(a,b,c,d,FeatureVector,mask,image):
    lenFV = len(FeatureVector)
    # currentFeature=M(i,arange())
    distance = float(1000)
    distance_now = float(1000)
    featurenum = - 1
    #........Search all the features for the closest occurance......................#
    for j in range(1, lenFV):
        #.......Fetching the current feature .........................................#
        CurSearch = FeatureVector[j]
        x = CurSearch.Position_X[-1]
        y = CurSearch.Position_Y[-1]
        x1 = a
        y1 = b
        #........Computing the distance between the current feature and its occurance in the past ....#
        distance_now = math.hypot(x1 - x, y1 - y)
        #........Compute the closest occurance of the feature point in the past........#
        if (distance_now < distance and distance_now < 70):
            distance = float(distance_now)
            featurenum = j
    #....Selected the closest feature in the past......................................#
    #.... featurenum stores the closest feature identity...............................#
    if (featurenum == - 1):
        #....If the featurenum is -1 that means closest feature is not found...........#
        Createfeature = Feature.Feature()
        #....Create feature object for this feature....................................#
        Createfeature.Identity = lenFV + 1
        featurenum = lenFV + 1
        #...Update the identity of the feature..........................................#
        Createfeature.UpdateFeatureOpticalFlow(a,c,b,d)
        #.......Update the optical flow of the feature..................................#
        FeatureVector.append(Createfeature)
    #......Append this feature to the FeatureVector array..........................#
    else:
        FeatureVector[featurenum].UpdateFeatureOpticalFlow(a,c,b,d)
        #.........The closest feature has identity featurenum, therefore uptate it with current image position (a,b,c,d)..#
        FeatureVector[featurenum].LinearFitFeature()
    #..........Linear fit the feature vector to derive the slope and intersept values ..#
    cv2.line(mask, (a, b), (c, d), color[featurenum].tolist(), 2)
    #.....For drawing on the output video update the mask with occurance of the feature and draw with the color allocated to this feature...#
    cv2.circle(image, (a, b), 5, color[featurenum].tolist(), -1)
    #.... Draw the circle to the current image indicating the current position of the feature in the frame. ................#
    cv2.putText(image, str(featurenum), (a, b), font, 1, color[featurenum].tolist(), 2, cv2.LINE_AA)
    # ....Draw the identity of the feature to the image..........................................#
    return [image,mask,FeatureVector]
#.......Return the image and masked which are we have just drawn and the FeatureVector which we updated............#

#.................If you are using the parser................................#
parser = OptionParser()
parser.add_option("-i", "--inputfile", dest="filename", help="write report to FILE", metavar="FILE")
(options, args) = parser.parse_args()
#............................................................................#

#.......Creating parser.........................................#
#print(__version__)
videoname = '../input.mp4'
#....... Input Video ...........................................#
cap = cv2.VideoCapture(videoname)
#......Create a video reader object from the file name..........#
width = int(cap.get(3))
height = int(cap.get(4))
FrameRate = cap.get(5)
#.....Getting the atributes of the video reader.................#

filename = '../outputVP1.avi'
#.....Output file ...........................................,,#

fourcc =  cv2.VideoWriter_fourcc('X','V','I','D')
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
#.......For some versions of OpenCV use the above line...........#
out = cv2.VideoWriter(filename, fourcc, FrameRate, (width,height))
#......Creating the video writer object........................#

color = np.random.randint(0,255,(5000,3))
# Create some random colors for coloring each of the feature point...#

font = cv2.FONT_HERSHEY_SIMPLEX
#.....Font for writing on the frame.............................#

#.......Forward vansihing point obect...........................#
VanishingPnt = VP.VP()
#.....Create the vanishing point object.........................#
VPdump = '../logVP1.csv'
VPfile = open(VPdump, 'w')
#.....Creating the log files...................................#

start = 3700
cap.set(1,start)
#......Go directly to the "start" numbered frame..............#
#......Usually pick the start when the vehicle is moving on a straight road..#
video_length = 8000
#......Until how long the video should run....................#
NumFrames= 15
#......Number of frames per each run..........................#
mu=1
while(count < video_length-1):
    mu=mu+1
    #.......For writing the path traced by each feature point for debugging in MATLAB..#
    folderName = str(os.path.splitext(videoname)[0])
    datadump = './lines/slope' + str(mu) + '.csv'
    slopefile = open(datadump, 'w')
    #...................................................................................#

    if (video_length-count<NumFrames):
        print(str(count)+','+ str(NumFrames))
        NumFrames=video_length-count-1
        if (video_length-count==0):
            break

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 500,qualityLevel = 0.3,minDistance = 70,blockSize = 10 )
    #........Defining the input parameters for the corner detection algorithm................#

    lk_params = dict( winSize  = (30,30),maxLevel = 3,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    #.......Defining parameters for tracking algorithm.......................................#

    ret, old_frame = cap.read()
    count =count+1
    #......Read the frame and increase the "count" (Counter) for keep track of the frames....#

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    #......Convert the frame to gray scale...................................................#

    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    #......Identify the good features to track ..............................................#

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    #..Update the vanishing point location to the mask to seek how the estimated vanishing point approaches the ground truth vanishing point..#
    mask = VanishingPnt.DrawVP(mask)

    #..Print which batch of the frames to STDOUT...........................................#
    print('Processing new set of frames'+str(mu))

    #... Create an empty vector for storing the features...................................#
    FeatureVector = []

    #....Intializing the Counter for the second "while-loop" 
    Counter=0
    #.....This while loop runs for "NumFrames" times and increments counter "Counter" every time it loops.
    while (Counter < NumFrames):
        #......Increment the counter for each itteration..................................#
        Counter=Counter+1

        ret,frame = cap.read()
        count =count+1
        #......Read the frame and increase the count......................................#

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #.....Conver the frame to gray scale..............................................#
        # calculate optical flow

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        #......Compute the optical flow...................................................#

        if p1 is None:
            break
        if p0 is None:
            break
        #.....If there are no feature points to track in the current frame or the future frame, we need to terminate this process and go to next set of 20 frames..........#

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        #.....Select the good features, which means status is "1"......................#

        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            #......Select the corresponding features from two fames....................#
            frame,mask, FeatureVector = UpdateFeatures(a,b,c,d,FeatureVector,mask,frame)
            #.....Update the features based on the closest distance from the past......#

            img = cv2.add(frame,mask)
            #.........Add the mask to the image for visualizations
            out.write(img)
            #........Write the imageot the video..........................................#

            '''
            For visualizing in live......................................................#
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            '''

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

#....Write all the feature points to the log file for analysis..................#
WriteLines(FeatureVector,slopefile)

#.....If Feature Vector is not None, then solve the vanishing point ............#
if FeatureVector is not None:
    VanishingPnt.FindVPfromFv(FeatureVector)
#......Finding the vanishing point from the featureVector...................#

VanishingPnt.SaveVPtoFile(VPfile)
#..........Write the vanishing point to the log file for analysis...............#

cv2.destroyAllWindows()
#.........Close all the windows if there is live visualizations.....................#

cap.release()
#..........Close the camera object..................................................#
