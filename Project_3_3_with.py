import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit

@jit(nopython=True)
def distance(r1,r2):
    r = r2-r1
    x = r[0]
    y = r[1]
    r12 = np.sqrt(x**2+y**2)
    return r12

@jit(nopython=True)
def f(r1,r2,r3,M2,M3): #use r1 as original
    G = 6.6738e-11
    M_j = 1.898e27
    fx = r1[2]
    fy = r1[3]
    fvx = G*(-M_j*r1[0]/distance(r1,0)**3 + M2*(r2-r1)[0]/distance(r1,r2)**3 + M3*(r3-r1)[0]/distance(r1,r3)**3)
    fvy = G*(-M_j*r1[1]/distance(r1,0)**3 + M2*(r2-r1)[1]/distance(r1,r2)**3 + M3*(r3-r1)[1]/distance(r1,r3)**3)
    return np.array([fx,fy,fvx,fvy])

@jit(nopython=True)
def RK(r1_0,r2_0,r3_0,M1,M2,M3):
    x1,y1 = [r1_0[0]],[r1_0[1]]
    x2,y2 = [r2_0[0]],[r2_0[1]]
    x3,y3 = [r3_0[0]],[r3_0[1]]
    r1, r2, r3 = np.copy(r1_0), np.copy(r2_0), np.copy(r3_0)
    t = 0
    h = 1 #1s
    j = 0
    period1 , period2, period3 = [],[],[]
    i = 1
    while  i >0:
       k11 = h*f(r1,r2,r3,M2,M3)
       k21 = h*f(r2,r1,r3,M1,M3)
       k31 = h*f(r3,r1,r2,M1,M2)
       
       k12 = h*f(r1+0.5*k11,r2+0.5*k21,r3+0.5*k31,M2,M3)
       k22 = h*f(r2+0.5*k21,r1+0.5*k11,r3+0.5*k31,M1,M3)
       k32 = h*f(r3+0.5*k31,r1+0.5*k11,r2+0.5*k21,M1,M2)
       
       k13 = h*f(r1+0.5*k12,r2+0.5*k22,r3+0.5*k32,M2,M3)
       k23 = h*f(r2+0.5*k22,r1+0.5*k12,r3+0.5*k32,M1,M3)
       k33 = h*f(r3+0.5*k32,r1+0.5*k12,r2+0.5*k22,M1,M2)
       
       k14 = h*f(r1+k13,r2+k23,r3+k33,M2,M3)
       k24 = h*f(r2+k23,r1+k13,r3+k33,M1,M3)
       k34 = h*f(r3+k33,r1+k13,r2+k23,M1,M2)
       
       r1 = r1 + (k11+2*k12+2*k13+k14)/6
       r2 = r2 + (k21+2*k22+2*k23+k24)/6
       r3 = r3 + (k31+2*k32+2*k33+k34)/6
       
       x1.append(r1[0]),y1.append(r1[1])
       x2.append(r2[0]),y2.append(r2[1])
       x3.append(r3[0]),y3.append(r3[1])
       t+=h
       j+=1
       #r1 is ganymede, use it as condition
       if y1[j-1]<0 and y1[j]>0:
           i = 0
           period1.append(t)
       if y2[j-1]<0 and y2[j]>0:
           period2.append(t)
       if y3[j-1]<0 and y3[j]>0:
           period3.append(t)
    return np.array([x1,y1]), period1, np.array([x2,y2]), period2, np.array([x3,y3]), period3


#initial conditons
Ganymede_0, Ganymede_M = np.array([1.069e9,0,0,10890], float), 6.42e23
Europa_0, Europa_M = np.array([6.648e8,0,0,13870], float), 5.97e24
Io_0, Io_M = np.array([4.201e8,0,0,17400], float), 8.93e22

Ganymede, p1, Europa, p2, Io, p3 = RK(Ganymede_0,Europa_0,Io_0,Ganymede_M,Europa_M,Io_M)
print("The orbital period of Ganymede is", p1[0]/86400, "days")
print("error =",abs(p1[0]/86400-7.155)/7.155*100,"%")
print("The orbital period of Europa is", p2[0]/86400, "days")
print("error =",abs(p2[0]/86400-3.551)/3.551*100,"%")
print("The orbital period of Io is", p3[0]/86400, "days")
print("error =",abs(p3[0]/86400-1.769)/1.769*100,"%")

#making animation
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='final project', artist='Matplotlib')
writer = FFMpegWriter(fps = 60, metadata = metadata )
fig,ax = plt.subplots()
line1, = plt.plot([],[],"o", color = "r", label= "Ganymede")
line2, = plt.plot([],[],"o", color = "g", label= "Europa")
line3, = plt.plot([],[],"o", color = "b", label= "Io")

# initialize the plot
ax.set_aspect("equal")
plt.xlim(-1.5e9,1.5e9)
plt.ylim(-1.5e9,1.5e9)
plt.plot([0],[0], 'o', color = "y", label="Jupiter")
ax.legend(loc = 1)
plt.xlabel("x (m)")
plt.ylabel("y (m)")


#saving each frame to make a mp4
with writer.saving(fig,"Animation_with.mp4",round(len(Ganymede[0])/1024)):
    for i in range(0,len(Ganymede[0]),1024): 
        line1.set_data([Ganymede[0,i]],[Ganymede[1,i]])
        line2.set_data([Europa[0,i]],[Europa[1,i]])
        line3.set_data([Io[0,i]],[Io[1,i]]) 
        writer.grab_frame()

    
    





