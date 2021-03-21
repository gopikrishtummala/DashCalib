import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from dateutil import parser
from pytz import timezone
import pytz

def FindClosestTimeIndex(CurTime,timelist):
    retlist =[]
    for i in range(0,len(timelist)):
        retlist.append(abs(timelist[i]-CurTime))
#    idx = (np.abs(timelist - CurTime)).argmin()
    return np.argmin(retlist)

eastern = timezone('US/Eastern')
gmt = pytz.timezone('GMT')
#................................................................#
#.......Read the GPS csv file..............#
df = pd.read_csv('2017-12-04_164442.csv', header=None)
timelist =[]
#.......Time list to store the time array....#
speedlist = []

#.......Speed list to store the time array....#
for i in range(1,len(df[4])):
    #.....Reading the time which is in the fourth column of the CSV file.
    cur = str(df[4][i])
    #.....Parsing the time to the standard format....#
    dt = parser.parse(cur)
    #.....Converting the time to micro-seconds format.
    CurTim = time.mktime(dt.timetuple()) * 1000
    #.....Reading the speed which is in the tenth column of the CSV file.
    CurSpeed = float(df[10][i])
    #.....Adding the time to the list
    timelist.append(CurTim)
    #.....Adding the speed to the list
    speedlist.append(CurSpeed)

#...Converting the speed to float object array...#
speedlist =np.array(speedlist, dtype=np.float32)
#................................................................#

#1512441890000
#.....Enter the video start time.....#
# Note: add 5 hours to the video creating time as the GPS file usually stores GMT time (which is 5 hours ahead of EST).
# Note: Please enter in YYYY-MM-DDTHH:MM:SS format
VideoStartTime ='2017-12-04T21:44:43'
#.. Parse the video time to the standard time..
dtVideo = parser.parse(VideoStartTime)
#... Frame rate of the video
framerate = 30
#...Video length in Minutes
VideoLenMin = 16
#...Rest of the seconds..
VideoLenSec = 18
#...... Enter the start frame number for velocity measurements...#
startFrame = 7189
endFrame = 7222

lengthofVideoSec = ((VideoLenMin*60)+VideoLenSec)*framerate
framelist =[]
speedlistFrame =[]
for i in range(0, lengthofVideoSec):
    curTime = (time.mktime(dtVideo.timetuple()) * 1000) + int(i*100/3)
    indx =FindClosestTimeIndex(curTime, timelist)
    #print(indx)
    curVel = speedlist[indx]
    #print(str(curTime)+','+str(curVel))
    framelist.append(curTime)
    speedlistFrame.append(curVel)


for i in range(startFrame,endFrame):
    print(speedlistFrame[i]*2.23694)
plt.plot(speedlistFrame)
plt.grid(True)
plt.ylabel('Velocity of the vehicle')
plt.show()


