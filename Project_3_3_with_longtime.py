import numpy as np
import matplotlib.pyplot as plt
from numba import jit

#calculate the angle of the planet
@jit(nopython=True) 
def angle(r1,r2): #use r1 as origin
    r = r2-r1
    x = r[0]
    y = r[1]
    if (x>=0 and y>=0):
        if x != 0: 
            theta = np.arctan(y/x)
        else:
            theta = np.pi/2
    elif (x<0 and y>=0):
        theta = np.pi - np.arctan(abs(y/x))
    elif (x<=0 and y<=0):
        if x != 0:
            theta = np.pi + np.arctan(abs(y/x))
        else:
            theta = 3*np.pi/2
    elif (x>0 and y<=0):
        theta = 2*np.pi - np.arctan(abs(y/x))
    return theta

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
    h = 450 #7.5mins
    j = 0
    period1 , period2, period3 = [],[],[]
    a = 1000*24*60*60 #1year
    t1,t2,t3 = 0,0,0
    while  t<a:
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
       t1+=h  #use to check period of indivial moons
       t2+=h
       t3+=h
       j+=1
       #r1 is ganymede, use it as condition
       if y1[j-1]<0 and y1[j]>0:
           period1.append(t1)
           t1 = 0
       if y2[j-1]<0 and y2[j]>0:
           period2.append(t2)
           t2 = 0
       if y3[j-1]<0 and y3[j]>0:
           period3.append(t3)
           t3 = 0
    return np.array([x1,y1]), period1, np.array([x2,y2]), period2, np.array([x3,y3]), period3


#initial conditons
Ganymede_0, Ganymede_M = np.array([1.069e9,0,0,10890], float), 6.42e23
Europa_0, Europa_M = np.array([6.648e8,0,0,13870], float), 5.97e24
Io_0, Io_M = np.array([4.201e8,0,0,17400], float), 8.93e22

Ganymede, p1, Europa, p2, Io, p3 = RK(Ganymede_0,Europa_0,Io_0,Ganymede_M,Europa_M,Io_M)
print("The average orbital period of Ganymede is", np.sum(p1)/len(p1)/86400, "days")
print("error =",abs(np.sum(p1)/len(p1)/86400-7.155)/7.155*100,"%")
print("The average orbital period of Europa is", np.sum(p2)/len(p2)/86400, "days")
print("error =",abs(np.sum(p2)/len(p2)/86400-3.551)/3.551*100,"%")
print("The average orbital period of Io is", np.sum(p3)/len(p3)/86400, "days")
print("error =",abs(np.sum(p3)/len(p3)/86400-1.769)/1.769*100,"%")

plt.plot(Ganymede[0],Ganymede[1], color = "r", label= "Ganymede")
plt.plot(Europa[0],Europa[1], color = "g", label= "Europa")
plt.plot(Io[0],Io[1], color = "b", label= "Io")
plt.plot([0],[0], 'o', color = "y", label="Jupiter")
plt.legend(loc = 1)
plt.xlabel("x (m)")
plt.ylabel("y (m)")


