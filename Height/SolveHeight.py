'''
------------------------------------------------------------------------------
--------------Part of AutoCalib project at The Ohio State University----------
--------------File Name: GroundTruth\ReprojectCalibration.py -----------------------------
Method to estimate the height of the camera.  

Usage -  ReprojectCalibration.py

--------------Authors: Gopi Krishna Tummala, Prasun Sinha -----
--------------Contact email: tummala.10@osu.edu, prasun@cse.ohio-state.edu
--------------For copyrights please contact the authors----------------------

------------------------------------------------------------------------------
ToDo:
------------------------------------------------------------------------------
'''
import cv2
import math
import numpy as np
from virtualPlane import drawVirtualPlanes
from numpy.linalg import inv
import csv

fx=(2510.23 / 3264) * 1920
fy=(2459.1 / 1836) * 1080
cx=(1626.09 / 3264) * 1920
cy=(977.91 / 1836) * 1080

cMtx = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1]], dtype=np.float32)
cMtx_inv = inv(cMtx)

def reprojectPoint3DHeight(point, rotM_inv, cMtx_inv, T, axConst, axis=1):
    tempMat = np.mat(rotM_inv) * np.mat(cMtx_inv) * np.mat(point)
    tempMat2 = np.mat(rotM_inv)* np.mat(T)
    #print "T\n\n"
    #print T
    #print "\n\ntempMat2\n\n"
    #print tempMat2
    s = axConst + tempMat2[axis,0]
    s /= tempMat[axis,0]

    ptProjection = (s*tempMat)-tempMat2
    return ptProjection

#http://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point?rq=1
#Default measuring plane = Z axis plane (2). x=0, y=1, z=2.
def reprojectPoint3D(point, rotM_inv, cMtx_inv, T, axConst, axis=1):
    tempMat = np.mat(rotM_inv) * np.mat(cMtx_inv) * np.mat(point)
    tempMat2 = np.mat(rotM_inv)* np.mat(T)
    #print "T\n\n"
    #print T
    #print "\n\ntempMat2\n\n"
    #print tempMat2
    s = axConst + tempMat2[axis,0]
    s /= tempMat[axis,0]

    ptProjection = (s*tempMat)-tempMat2
    return ptProjection

def ComputeDefZ(pnt1,pnt2):
    return (pnt2[2]-pnt1[2])
def ComputeReprojectionErrorZ(ZRPoints):
    length = len(ZRPoints)
    dist =[]
    err = 0
    '''
    for i in range(0,length-1):
        CurDist = ComputeDefZ(ZRPoints[0], ZRPoints[i+1])
        dist.append(CurDist)
    '''
    CurDist = ComputeDefZ(ZRPoints[0], ZRPoints[5])
    err = abs(CurDist-5)/5
    return err
def ComputeAvgVelocity(hPntArray):
    totalMov=0
    SpeedArray =[]
    for i in range(0,len(hPntArray)-1):
        CurMv = ComputeDefZ(hPntArray[i+1], hPntArray[i])
        totalMov = totalMov + CurMv
        CurSpeed = abs(CurMv)*30*2.23694
        SpeedArray.append(CurSpeed)
    avgMv = totalMov/(len(hPntArray)-1)
    Speed = avgMv*30*2.23694
    return SpeedArray

TVecg = np.array([[-0.71910787],[1.90039921],[5.54637718]],dtype=np.float32)
RVecg = np.array([[-0.151899],[0.1455037],[0.0177547]],dtype=np.float32)

TVech = np.array([[0.0],[0.0],[0.0]],dtype=np.float32)
RMatg,Rj1 = cv2.Rodrigues(RVecg)
invRMatg = inv(RMatg)
print(invRMatg)
RMat = np.array([[0.9864,-0.0369,0.1600],[0.0154,0.9909,0.1335],[-0.1635,-0.1293,0.9780]],dtype=np.float32)
invRMatE = inv(RMat)
print(invRMatE)

RVecE,Rj = cv2.Rodrigues(RMat)
XIPoints =[]
ZRPoints =[]
ZRPointsE =[]
hPntArray = []
IPos =[]#np.array([], dtype=np.float32)
RPos =[]#np.array([], dtype=np.float32)
with open('./annotated/lightpoints.csv') as csvfile:
    reader = csv.reader(csvfile)
    count  = 0
    for row in reader:
        p1A = np.array([row[0], row[1]], dtype=np.float32)
        pnt = np.array([[row[0]], [row[1]],[1.0]], dtype=np.float32)
        curPnt =reprojectPoint3D(pnt, invRMatg, cMtx_inv, TVecg, 0)
        curPntE = reprojectPoint3D(pnt, invRMatE, cMtx_inv, TVecg, 0)
        hPnt =  reprojectPoint3DHeight(pnt, invRMatE, cMtx_inv, TVech, 1.4)
        hPntArray.append(hPnt)
        ZRPointsE.append(curPntE)
        ZRPoints.append(curPnt)
        print(hPnt)
        XIPoints.append(p1A)

        '''
        IPos = np.append(IPos,[p1A])
        IPos = np.append(IPos,[p1B])
        IPos = np.append(IPos,[p1C])
        IPos = np.append(IPos,[p1D])
        '''
        r1A =np.array([0,0,count], dtype=np.float32)
        r1B =np.array([0.57,0,count], dtype=np.float32)
        r1C =np.array([0,-0.5,count], dtype=np.float32)
        r1D =np.array([0.57,-0.5,count], dtype=np.float32)
        RPos.append(r1A)
        RPos.append(r1B)
        RPos.append(r1C)
        RPos.append(r1D)
#print(ComputeReprojectionErrorZ(ZRPoints),ComputeReprojectionErrorZ(ZRPointsE),ComputeReprojectionErrorZ(ZRPointsE))

#...........Adjust the height until this velocity matches the velocity from GPS
print((ComputeAvgVelocity(hPntArray)))