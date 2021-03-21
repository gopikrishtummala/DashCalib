import numpy as np
from scipy import stats
import cv2
import math
from sympy import *
from sympy.solvers import solve
#s,r11,r12,r13,r21,r22,r23 = symbols('s r11 r12 r13 r21 r22 r23')
'''
fx=(2510.23 / 3264) * 1920
fy=(2459.1 / 1836) * 1080
cx=(1626.09 / 3264) * 1920
cy=(977.91 / 1836) * 1080
'''
fx=(2892.2/ 3264) * 1920
fy=(2758.0 / 1836) * 1080
cx=(1661.1 / 3264) * 1920
cy=(1228.1 / 1836) * 1080
fx=(2510.23 / 3264) * 3840
fy=(2459.1 / 1836) * 2160
cx=(1626.09 / 3264) * 3840
cy=(977.91 / 1836) * 2160
#cx = 1920/2
#cy = 1080/2
#vp1X = 1110.00
#vp1Y = 821.00
vp1X = 2143.56
vp1Y = 1653.48

r33 = 1/math.sqrt(1+ (((vp1X-cx)/fx)**2)+(((vp1Y-cy)/fy)**2))
r13 = ((vp1X-cx)/fx)*r33
r23 = ((vp1Y-cy)/fy)*r33
#... This assumes angle around x-axis is 0...#
#yangle  = math.atan2(r13,r33)
#zangle =  math.asin(r23)
#... This assumes angle around x-axis is 0...#

#print(r13,r23,r33,yangle,zangle)
r21 =0
r31 =0
r22 = math.sqrt(1-(r23**2))
r32 = -math.sqrt(1-(r33**2))
r12 = 0#math.sqrt(1-(r13**2))
#r11 = math.sqrt(1-(r13**2)-(r12**2))
r11 = 1
print(r11,r12,r13)
print(r21,r22,r23)
print(r31,r32,r33)
R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]], dtype=np.float32)
zangle = math.atan(r32/r33)

yangle = math.acos(r33/cos(zangle))
print(zangle,yangle)
'''
vp2X = 10000.00
vp2Y = 500.00
r32 = 1/math.sqrt(1+ (((vp2X-cx)/fx)**2)+(((vp2Y-cy)/fy)**2))
r12 = ((vp2X-cx)/fx)*r32
r22 = ((vp2Y-cy)/fy)*r32
print(r12,r22,r32)

r33 = math.sqrt(1-(r32**2)-(r31**2))
r23 = math.sqrt(1-(r22**2)-(r21**2))
#print(r33,r23)
r13 = math.sqrt(1-(r33**2)-(r23**2))
print(r13,r23,r33)
'''