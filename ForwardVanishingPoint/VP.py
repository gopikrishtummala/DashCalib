'''
------------------------------------------------------------------------------
--------------Part of DashCalib project at The Ohio State University----------
--------------File Name: VP.Py-----------------------------
Implements the vanishing point class

--------------Authors: Gopi Krishna Tummala, Prasun Sinha and, Tanmoy Das-----
--------------Contact email: tummala.10@osu.edu, prasun@cse.ohio-state.edu
--------------For copyrights please contact the authors----------------------

------------------------------------------------------------------------------
ToDo:
------------------------------------------------------------------------------
'''
import numpy as np
import cv2
class VP:
	
    def __init__(self):
        #...........For storing the X-Coordinates of the vanishing point.......#
        self.X = []
        #...........For storing the Y-Coordinates of the vanishing point.......#
        self.Y = []
        #...........Current estimate of the X-Coordinates of the vanishing point.......#
        self.Xval = 0
        #...........Current estimate of the Y-Coordinates of the vanishing point.......#
        self.Yval = 0

    def FindVPfromFv(Self, FeatureVector):
        # We estimate the vanishing points based on the intersections of feature point tracks.......# 
		#..... Array for storing the X-Coordinates of the line intersections.......#
        xArray = []
        #..... Array for storing the Y-Coordinates of the line intersections.......#
        yArray = []
        for i in range(0, len(FeatureVector)):
            for j in range(0, len(FeatureVector)):
                #....... Slope of the line along the track of feature point-1 ..........................#
		m1 = FeatureVector[i].Mval
                #....... Slope of the line along the track of feature point-2 ..........................#
                m2 = FeatureVector[j].Mval
                #....... Interscept of the line along the track of feature point-1 ..........................#
                c1 = FeatureVector[i].Cval
                #....... Interscept of the line along the track of feature point-2 ..........................#
                c2 = FeatureVector[j].Cval
                #........Estimating the intersection only if the lines are sufficently long, i.e., length greater than 15...#
                if m1 != m2 and m1 != 0 and m2 != 0 and len(FeatureVector[i].Position_X)>15 and len(FeatureVector[j].Position_X)>15:
                    xVal = float(c1 - c2) / float(m2 - m1)
                    yVal = (float(m1) * float(xVal)) + float(c1)
                    #.........Eppending the intersections to the xArray and yArray
                    xArray.append(xVal)
                    yArray.append(yVal)

	#.......Solving the median of intersection points for estimating the vanishing points...........#
        if xArray is not None and yArray is not None:
            vpcurX =np.median(xArray)
            vpcurY =np.median(yArray)
        
	if 1==1:#vpcurX < 1100 and vpcurX > 700 and vpcurY > 500:
	    #...If the length of vanishing point vector is less than 10 keep estimating more......#
            if len(Self.X)<10:
                Self.X.append(vpcurX)
                Self.Y.append(vpcurY)
                if Self.Xval is not None and Self.Yval is not None:
                    Self.Xval = np.median(Self.X)
                    Self.Yval = np.median(Self.Y)
                    print (Self.Xval,Self.Yval)

	    #.....For removing the outliers, if the estimated vanishing point is far from the past estimates remove it as it might be outlier...#
            elif abs(vpcurX-Self.Xval)<110 and abs(vpcurY-Self.Yval)<110:
                Self.X.append(vpcurX)
                Self.Y.append(vpcurY)
                Self.Xval = np.median(Self.X)
                Self.Yval = np.median(Self.Y)
                print (Self.Xval,Self.Yval)

    #......Method to draw the vanishing point from Ground truth estimation, which will be useful for debugging ..# 
    def DrawVP(Self,mask):
        if Self.Xval is not None and Self.Yval is not None:
            if Self.Xval!=0 and Self.Yval!=0:
                cv2.circle(mask, (int(Self.Xval), int(Self.Yval)), 10, (0, 255, 255), -1)
                cv2.circle(mask, (int(1080), int(701)), 10, (255, 0, 255), -1)
        return mask

    #.......Method to save the current vanishing point to the log file. 
    def SaveVPtoFile(self,file):
        file.write(str(self.Xval) + ',' + str(self.Yval)+'\n')
