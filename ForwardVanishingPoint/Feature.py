'''
------------------------------------------------------------------------------
--------------Part of DashCalib project at The Ohio State University----------
--------------File Name: Feature.Py-----------------------------
Implements the Feature point class

--------------Authors: Gopi Krishna Tummala, Prasun Sinha and, Tanmoy Das-----
--------------Contact email: tummala.10@osu.edu, prasun@cse.ohio-state.edu
--------------For copyrights please contact the authors----------------------

------------------------------------------------------------------------------
ToDo:
------------------------------------------------------------------------------
'''

from scipy import stats
class Feature:
    # Class of the features in
    def __init__(self):
	#........Useful parameters..................#
	self.Position_X = [];
        self.Position_Y = [];

	#......To store the line parameters that are fitted along the path traced by the feature point..#
        self.Mval = 0;
	#......Slope value..........#
        self.Cval = 0;
	#......Intersept value..........#	
        self.time = [];
	#......To store the absolute time ...#
        self.Frame_Number = [];
	#......To store the frame number ...#
        self.Color = [];
	#......To store the color ...#
        self.Identity = 500;
	#......To store the identity of the feature point ...#

	#.....For future use........................#
	self.Realworld_PosX = [];
        self.Realworld_PosY = [];
        self.angle_X = [];
        self.angle_Y = [];
        self.Angular_Velocity_X = [];
        self.Angular_Velocity_Y = [];
        self.MovAngle_X = [];
        self.MovAngle_Y = [];
        self.MovingObject = 0;
        self.Position_Z = [];
        self.MovementX = [];
        self.MovementY = [];
        self.StartDist = 0;

    #......Method to update the feature point positions.........#
    def UpdateFeatureOpticalFlow(self, a, c, b, d):
        self.Position_X.append(a)
        self.Position_X.append(c)
        self.Position_Y.append(b)
        self.Position_Y.append(d)

    #......Method to fit lines along the track of the feature point .........#
    def LinearFitFeature(self):
        if len(self.Position_Y) > 10:
            self.Mval, self.Cval, r_value, p_value, std_err = stats.linregress(self.Position_X,self.Position_Y)
