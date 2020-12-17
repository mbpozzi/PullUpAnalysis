# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:32:17 2020

@author: Michael
"""
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import cv2 as cv 
import time
import math
import scipy

# Functions
def linepoint(img, heelx, heely, toex, toey, cm_x, cm_y, line_col, line_thick, circle_col, circle_thick, rad, index):
    img = cv.line(img, (int(heelx[index]), int(heely[index])), (int(toex[index]), int(toey[index])), line_col, line_thick)
    img = cv.circle(img, (int(cm_x[index]), int(cm_y[index])), rad, circle_col, circle_thick)
    return img

def angle(img, vx, vy, pt1x, pt1y, pt2x, pt2y, index):
    vect1x = pt1x[index] - vx[index]
    vect1y = pt1y[index] - vy[index]
    vect2x = pt2x[index] - vx[index]
    vect2y = pt2y[index] - vy[index]
    dot = vect1x*vect2x + vect1y*vect2y
    mag1 = pow(vect1x,2) + pow(vect1y,2)
    mag2 = pow(vect2x,2) + pow(vect2y,2)
    mag1 = math.sqrt(mag1)
    mag2 = math.sqrt(mag2)
    angle = np.arccos(dot/(mag1*mag2))
    angle = int(math.degrees(angle))
    #print(angle)
    img = cv.putText(img, str(angle), (int(vx[index]), int(vy[index])), cv.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
    return img

def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError 
    if x.size < window_len:
        raise ValueError 
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError 
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

# load files derived from DLTdv8 and xls or xlsx files recieved from class
data_file2 = 'tbcm.xls'
data_file = 'temp_flip.xls'

#Load Data 
applet = pd.read_excel(data_file, "applet data")
scale_factor = pd.read_excel(data_file, "scale factor")
foot = pd.read_excel(data_file, "right foot")
shank = pd.read_excel(data_file, "right shank")
thigh = pd.read_excel(data_file, "right thigh")
trunk = pd.read_excel(data_file, 'trunk')
hand = pd.read_excel(data_file, 'right hand')
forearm = pd.read_excel(data_file, 'right forearm')
arm = pd.read_excel(data_file, 'right upper arm')
head = pd.read_excel(data_file, 'head')
tbcm = pd.read_excel(data_file, "TBCM")
speed = pd.read_excel(data_file2, "TBCM")

#TBCM
cm_x = tbcm["TBCM in x"]
cm_y = tbcm["TBCM in y"]
uncm_x = np.array(cm_x)
uncm_y = np.array(cm_y)

#foot
toex = applet["TOEX"]
toex = np.array(toex)[3:107]
toey = applet["TOEY"]
toey = np.array(toey)[3:107]
heelx = applet["HEELX"]
heely = applet["HEELY"]
heelx = np.array(heelx)[3:107]
heely = np.array(heely)[3:107]
footcm_x = foot["CM in x [Px+CM%* Dx]"]
footcm_y = foot["CM in y [Py+CM%* Dy]"]
footcm_x = np.array(footcm_x)[2:106]
footcm_y = np.array(footcm_y)[2:106]

        
#shank
anklex = applet["ANKLEX"]
ankley = applet["ANKLEY"]
kneex = applet["KNEEX"]
kneey= applet['KNEEY']
anklex = np.array(anklex)[3:107]
ankley = np.array(ankley)[3:107]
kneex = np.array(kneex)[3:107]
kneey = np.array(kneey)[3:107]
shankcm_x = shank["CM in x [Px+CM%* Dx]"]
shankcm_y = shank["CM in y [Py+CM%* Dy]"]
shankcm_x = np.array(shankcm_x)[2:106]
shankcm_y = np.array(shankcm_y)[2:106]

#thigh 
hipx = applet["HIPX"]
hipy = applet["HIPY"]
hipx = np.array(hipx)[3:107]
hipy = np.array(hipy)[3:107]
thighcm_x = thigh["CM in x [Px+CM%* Dx]"]
thighcm_y = thigh["CM in y [Py+CM%* Dy]"]
thighcm_x = np.array(thighcm_x)[2:106]
thighcm_y = np.array(thighcm_y)[2:106]

#trunk
c7x = applet["C7X"]
c7y = applet["C7Y"]
c7x = np.array(c7x)[3:107]
c7y = np.array(c7y)[3:107]
trunkcm_x = trunk["CM in x [Px+CM%* Dx]"]
trunkcm_y = trunk["CM in y [Py+CM%* Dy]"]
trunkcm_x = np.array(trunkcm_x)[2:106]
trunkcm_y = np.array(trunkcm_y)[2:106]

#upper arm 
shoulderx = applet["SHOULDERX"]
shouldery = applet["SHOULDERY"]
shoulderx = np.array(shoulderx)[3:107]
shouldery = np.array(shouldery)[3:107]
elbowx = applet["ELBOWX"]
elbowy = applet["ELBOWY"]
elbowx = np.array(elbowx)[3:107]
elbowy = np.array(elbowy)[3:107]
armcm_x = arm["CM in x [Px+CM%* Dx]"]
armcm_y = arm["CM in y [Py+CM%* Dy]"]
armcm_x = np.array(armcm_x)[2:106]
armcm_y = np.array(armcm_y)[2:106]

#forearm
wristx = applet["WRISTX"]
wristy = applet["WRISTY"]
wristx = np.array(wristx)[3:107]
wristy = np.array(wristy)[3:107]
forearmcm_x = forearm["CM in x [Px+CM%* Dx]"]
forearmcm_y = forearm["CM in y [Py+CM%* Dy]"]
forearmcm_x = np.array(forearmcm_x)[2:106]
forearmcm_y = np.array(forearmcm_y)[2:106]

#hand 
fingerx = applet["FINGERX"]
fingery = applet["FINGERY"]
fingerx = np.array(fingerx)[3:107]
fingery = np.array(fingery)[3:107]
handcm_x = hand["CM in x [Px+CM%* Dx]"]
handcm_y = hand["CM in y [Py+CM%* Dy]"]
handcm_x = np.array(handcm_x)[2:106]
handcm_y = np.array(handcm_y)[2:106]

#head 
vertexx = applet["VERTEXX"]
vertexy = applet["VERTEXY"]
vertexx = np.array(vertexx)[3:107]
vertexy = np.array(vertexy)[3:107]
headcm_x = head["CM in x [Px+CM%* Dx]"]
headcm_y = head["CM in y [Py+CM%* Dy]"]
headcm_x = np.array(headcm_x)[2:106]
headcm_y = np.array(headcm_y)[2:106]

#speed 
time = speed["time"]
velx = speed["Velocity in x"]
vely = speed["Velocity in y"]
tbcm_mx = speed["ChangeX"]
tbcm_my = speed["ChangeY"]

smoothy = smooth(vely, 8, 'blackman')
smoothx = smooth(velx, 8, 'blackman')
smoothy = smoothy[:104]
smoothx = smoothx[:104]

#Estimated force array created from values  
rfv = []
for i in range(len(velx)):
    if i < 26:
        rfv.append(775)
    elif i < 40:
        rfv.append(-33.9286*i + 1657.1436)
    elif i < 54:
        rfv.append(33.9286*i - 1057.144)
    elif i < 67:
        rfv.append(41.9231*i - 1488.8474)
    else:
        rfv.append(-15.5714*i + 2363.2838)

rfh = []
for x in range(len(velx)):
    if x < 21:
        rfh.append(0)
    elif x < 38:
        rfh.append((380/17)*x - (7980/17))
    elif x < 54:
        rfh.append((-95/4)*x + (2565/2))
    elif x < 72: 
        rfh.append((-200/9)*x + (1200))
    elif x < 89:
        rfh.append((400/17)*x - (35600/17))
    else:
        rfh.append((100/13)*x - (8900/13))
        
#Angular 
comx = speed["TBCM in x"]
comy = speed["TBCM in y"]
copx = 492.99519
copy = 1920-461.895996
copx = copx * .001859
copy = copy * .001859
dx = []
dy = []
for i in range(len(comx)):
    dx.append(copx-comx[i])
    dy.append(comy[i]-copy)
plt.plot(dx)
plt.plot(dy)
summ = []
for i in range(len(vely)):
    if i < 21:
        summ.append((10/7)*i + 50)
    elif i < 26:
        summ.append((-16)*i + (416))
    elif i < 38:
        summ.append(-25*i + 650)
    elif i < 63:
        summ.append((12)*i - (756))
    elif i < 72:
        summ.append((175/9)*i - 1225)
    else:
        summ.append((-13/3)*i + 487)

Mv = []
Mh = []        
Ms = summ
for i in range(len(vely)):
    Mv.append(rfv[i]*dx[i])    
    Mh.append(rfh[i]*dy[i])
    #Ms.append((rfv[i]*dx[i])+(rfh[i]*dy[i]))
# Angular Impulse Graphs
plt.figure()
plt.subplot(211)
plt.plot(time, dx, label='Horizantal Moment Arm')
plt.plot(time, dy, label='Vertical Moment Arm')
plt.grid(True)
plt.legend()
plt.title("Moment Arm Lengths vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Length (m)")
plt.subplot(212)
plt.plot(time, Mv, label="Moment from RFv")
plt.plot(time, Mh, label="Moment from RFh")
plt.plot(time, Ms, label="Estimated Moment Sum")
plt.fill_between(time, Ms)
plt.grid(True)
plt.title("Estmated Moments vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Moment (Nm)")
plt.legend()
plt.show()

# Position, Velocity, Reaction force graphs 
plt.subplot(311)
plt.plot(time, tbcm_mx, label="Horizontal CM")
plt.plot(time, tbcm_my, label="Vertical CM")
plt.title("CM vs Time")
plt.ylabel("Change in Position (m)")
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True)
plt.subplot(312)
plt.plot(time, smoothx, label="Horizontal Vel")
plt.plot(time, smoothy, label="Vertical Vel")
plt.title("Velocity vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Velcity (m/s)")
plt.legend()
plt.grid(True)
plt.subplot(313)
plt.plot(time, rfv, label="RFv")
plt.plot(time, rfh, label="RFh")
plt.plot(time, [-775]*len(time), label = "BW")
plt.title("Estimated Reaction Force vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.grid(True)
plt.legend()

#Reaction force and impulse graphs 
plt.subplot(211)
plt.plot(time, rfv, label="RFv")
#plt.fill_between(time, rfv)
plt.plot(time, rfh, label="RFh")
#plt.fill_between(time, rfh)
#plt.plot(time, [-775]*len(time), label = "BW")
#plt.fill_between(time, [-775]*len(time))
plt.title("Estimated Reaction Force vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.grid(True)
plt.legend()
plt.subplot(212)

plt.plot(time, rfv, label="RFv")
#plt.plot(time, rfh, label="RFh")
plt.plot(time, [-775]*len(time), label = "BW")
plt.fill_between(time, rfv)
#plt.fill_between(time, rfh)
plt.fill_between(time, [-775]*len(time))

plt.title("Estimated Reaction Force vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.grid(True)
plt.legend()
plt.show()

#Shulder/Elbow Angular Analysis 
ang = pd.read_excel("angleangleexcel.xlsx")
sh = ang["Shoulder"]
elb = ang["Elbow"]
fig, ax = plt.subplots()
plt.scatter(sh, elb, linewidths=3)
plt.scatter(51, 76, linewidths=3, label = "Start")
plt.scatter(67, 101, linewidths = 3, label = "RFvMin")
plt.scatter(134, 140, linewidths = 3, label = "RFvMax")
plt.scatter(4,73, linewidths = 3, label = "Release")
plt.xlim(0,160)
plt.ylim(0,160)
ax.set_aspect("equal")
ax.grid(b=True, which='major', color='k', linestyle='--')
ax.set_xlabel('Shoulder Angle', fontsize=14)
ax.set_ylabel('Elbow Angle', fontsize=14)
ax.set_title('Shoulder vs Elbow Angle', fontsize=18)
plt.legend()
plt.show()

ang = pd.read_excel("angleangleexcel.xlsx")
sh = ang["sh pu"]
elb = ang["elb pu"]
fig, ax = plt.subplots()
ax.set_aspect("equal")
plt.scatter(sh, elb, linewidths=3)
plt.scatter(170, 172, linewidths=3, label = "Start")
plt.scatter(123, 145, linewidths = 3, label = "RFv Max")
plt.scatter(81, 106, linewidths = 3, label = "BW Crossing")
plt.scatter(53,77, linewidths = 3, label = "Min RFv")
plt.scatter(6,36, linewidths = 3, label = "End")
plt.xlim(0,180)
plt.ylim(0,180)
ax.grid(b=True, which='major', color='k', linestyle='--')
ax.set_xlabel('Shoulder Angle', fontsize=14)
ax.set_ylabel('Elbow Angle', fontsize=14)
ax.set_title('Shoulder vs Elbow Angle', fontsize=18)
plt.legend()
plt.show()

ang = pd.read_excel("angleangleexcel.xlsx")
sh = ang["sh exp"]
elb = ang["elb exp"]
fig, ax = plt.subplots()
ax.set_aspect("equal")
plt.scatter(sh, elb, linewidths=3)
plt.scatter(145,163, linewidths=3, label = "Start")
plt.scatter(120, 147, linewidths = 3, label = "RFv Max")
plt.scatter(9, 88, linewidths = 3, label = "End")
plt.xlim(0,180)
plt.ylim(0,180)
ax.grid(b=True, which='major', color='k', linestyle='--')
ax.set_xlabel('Shoulder Angle', fontsize=14)
ax.set_ylabel('Elbow Angle', fontsize=14)
ax.set_title('Shoulder vs Elbow Angle', fontsize=18)
plt.legend()
plt.show()

#Trunk/Leg Analysis 
trunkang = []
for x in range(len(anklex)):
    vect1x = trunkcm_x[x] - hipx[x]
    vect1y = trunkcm_y[x] - hipy[x]
    vect2x = 100
    vect2y = 0
    dot = vect1x*vect2x + vect1y*vect2y
    mag1 = pow(vect1x,2) + pow(vect1y,2)
    mag2 = pow(vect2x,2) + pow(vect2y,2)
    mag1 = math.sqrt(mag1)
    mag2 = math.sqrt(mag2)
    angle = np.arccos(dot/(mag1*mag2))
    angle = int(math.degrees(angle))
    trunkang.append(angle)

legang = []
for x in range(len(anklex)):
    vect1x = thighcm_x[x] - hipx[x]
    vect1y = thighcm_y[x] - hipy[x]
    vect2x = 100
    vect2y = 0
    dot = vect1x*vect2x + vect1y*vect2y
    mag1 = pow(vect1x,2) + pow(vect1y,2)
    mag2 = pow(vect2x,2) + pow(vect2y,2)
    mag1 = math.sqrt(mag1)
    mag2 = math.sqrt(mag2)
    angle = np.arccos(dot/(mag1*mag2))
    angle = int(math.degrees(angle))
    legang.append(angle)

elbowang = []
for x in range(len(anklex)):
    vect1x = forearmcm_x[x] - elbowx[x]
    vect1y = forearmcm_y[x] - elbowy[x]
    vect2x = 100
    vect2y = 0
    dot = vect1x*vect2x + vect1y*vect2y
    mag1 = pow(vect1x,2) + pow(vect1y,2)
    mag2 = pow(vect2x,2) + pow(vect2y,2)
    mag1 = math.sqrt(mag1)
    mag2 = math.sqrt(mag2)
    angle = np.arccos(dot/(mag1*mag2))
    angle = int(math.degrees(angle))
    elbowang.append(angle)
    
upperarmang = []
for x in range(len(anklex)):
    vect1x = armcm_x[x] - shoulderx[x]
    vect1y = armcm_y[x] - shouldery[x]
    vect2x = 100
    vect2y = 0
    dot = vect1x*vect2x + vect1y*vect2y
    mag1 = pow(vect1x,2) + pow(vect1y,2)
    mag2 = pow(vect2x,2) + pow(vect2y,2)
    mag1 = math.sqrt(mag1)
    mag2 = math.sqrt(mag2)
    angle = np.arccos(dot/(mag1*mag2))
    angle = int(math.degrees(angle))
    upperarmang.append(angle)
    
fig, ax = plt.subplots()
plt.scatter(legang, trunkang, linewidths=3)
plt.scatter(85,96, linewidths=3, label = "Start")
plt.scatter(67, 116, linewidths = 3, label = "RFv Max")
plt.scatter(6, 112, linewidths = 3, label = "End")
plt.xlim(0,180)
plt.ylim(0,180)
ax.set_aspect("equal")
ax.grid(b=True, which='major', color='k', linestyle='--')
ax.set_xlabel('Leg Angle', fontsize=14)
ax.set_ylabel('Trunk Angle', fontsize=14)
ax.set_title('Leg vs Trunk Angle', fontsize=18)
plt.legend()
plt.show()

plt.figure()
plt.plot(time, tbcm_mx, label="Horizantal CM")
plt.plot(time, tbcm_my, label="Vertical CM")
plt.title("CM vs Time")
plt.ylabel("Change in Position (m)")
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True)


#VIDEO
video_name = "60fps.mp4"
#set frame
cap = cv.VideoCapture(video_name)
frame = 3985
end = 4100
cap.set(cv.CAP_PROP_POS_FRAMES, frame)

#set line parameters
l_color = (0,0,255)
l_thick = 5
c_color = (255,255,255)
c_rad = 5
c_thick = 5
radius = 20
color = (30,255,255)
thickness = -1

#set indexes
index = 0
#set save values 
file = "cm_pullup_skel2.mp4"
#out = cv.VideoWriter(file,0x7634706d, cap.get(cv.CAP_PROP_FPS), (1080, 1920))	#Set videowriter
while(True):
    ret, img = cap.read()
    if index < 104:
        img = cv.circle(img, (int(uncm_x[index]), int(uncm_y[index])), radius, color, thickness)
        img = linepoint(img, heelx, heely, toex, toey, footcm_x, footcm_y, l_color, l_thick, c_color, c_thick, c_rad, index)
        img = linepoint(img, anklex, ankley, kneex, kneey, shankcm_x, shankcm_y, l_color, l_thick, c_color, c_thick, c_rad, index)
        img = linepoint(img, hipx, hipy, kneex, kneey, thighcm_x, thighcm_y, l_color, l_thick, c_color, c_thick, c_rad, index)
        img = linepoint(img, hipx, hipy, c7x, c7y, trunkcm_x, trunkcm_y, l_color, l_thick, c_color, c_thick, c_rad, index)
        img = linepoint(img, elbowx, elbowy, shoulderx, shouldery, armcm_x, armcm_y, l_color, l_thick, c_color, c_thick, c_rad, index)
        img = linepoint(img, elbowx, elbowy, wristx, wristy, forearmcm_x, forearmcm_y, l_color, l_thick, c_color, c_thick, c_rad, index)
        img = linepoint(img, fingerx, fingery, wristx, wristy, handcm_x, handcm_y, l_color, l_thick, c_color, c_thick, c_rad, index)
        img = linepoint(img, vertexx, vertexy, c7x, c7y, headcm_x, headcm_y, l_color, l_thick, c_color, c_thick, c_rad, index)
        img = angle(img, shoulderx, shouldery, hipx, hipy, elbowx, elbowy, index)
        img = angle(img, elbowx, elbowy, shoulderx, shouldery, wristx, wristy, index)
        img = angle(img, wristx, wristy, elbowx, elbowy, fingerx, fingery, index)
        if index < 104:
            numx = truncate(velx[index], 3)
            numy = truncate(vely[index], 3)
            numx = "Vel x [m/s]: " + str(numx)
            numy = "Vel y [m/s]: " + str(numy)
            img = cv.putText(img, numx, (30,60), cv.FONT_HERSHEY_SIMPLEX, 2, (209, 80, 0, 255), 3)
            img = cv.putText(img, numy, (30,140), cv.FONT_HERSHEY_SIMPLEX, 2, (209, 80, 0, 255), 3)
    if ret == True:
        #cv.imshow("frame", img)
        #out.write(img)		
        if cv.waitKey(25) & 0xFF == ord('q'): 
            break
        frame += 1
        index += 1
        
        if frame == end:
            break
#out.release()
cap.release()
cv.destroyAllWindows()




        

    


