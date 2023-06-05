import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numba import jit
import time


@jit(nopython=True)
def distance(r1,r2):
    r = r2 -r1
    x = r[0]
    y = r[1]
    z = r[2]
    r12 = np.sqrt(x**2+y**2+z**2)
    return r12

#the function for the effect of star 2 and 3 to 1
@jit(nopython=True)
def f(r1,r2,r3):
    G = 6.6738e-11
    Msun = 1.9891e30
    r12 = distance(r1,r2)
    r13 = distance(r1,r3)
    fx = r1[3]
    fy = r1[4]
    fz = r1[5]
    #x,y,z component of the force using angle
    #take r1 as origin
    fvx = G*Msun*((r2-r1)[0]/r12**3 + (r3-r1)[0]/r13**3)
    fvy = G*Msun*((r2-r1)[1]/r12**3 + (r3-r1)[1]/r13**3)
    fvz = G*Msun*((r2-r1)[2]/r12**3 + (r3-r1)[2]/r13**3)
    return np.array([fx,fy,fz,fvx,fvy,fvz])

@jit(nopython=True)
def star_pos_RK(sun1_0,sun2_0,sun3_0):
    x1,y1,z1 = [sun1_0[0]],[sun1_0[1]],[sun1_0[2]]
    x2,y2,z2 = [sun2_0[0]],[sun2_0[1]],[sun2_0[2]]
    x3,y3,z3 = [sun3_0[0]],[sun3_0[1]],[sun3_0[2]]
    sun1, sun2, sun3 = np.copy(sun1_0),np.copy(sun2_0),np.copy(sun3_0)
    h = 900 #15 mins
    t = 0
    a = 3000000 #h*a is the total time
    while t < h*a: #max no. of day: 31250
        # Asumme their radius also like our sun, stop it when they are too close
        if distance(sun1,sun2)< 2*7e8 or distance(sun1,sun3)< 2*7e8 or distance(sun2,sun3)< 2*7e8:
            print("Collided!!")
            return [x1,y1,z1],[x2,y2,z2],[x3,y3,z3],t
        k11 = h*f(sun1,sun2,sun3)
        k21 = h*f(sun2,sun1,sun3)
        k31 = h*f(sun3,sun1,sun2)
        
        k12 = h*f(sun1+0.5*k11,sun2+0.5*k21,sun3+0.5*k31)
        k22 = h*f(sun2+0.5*k21,sun1+0.5*k11,sun3+0.5*k31)
        k32 = h*f(sun3+0.5*k31,sun1+0.5*k11,sun2+0.5*k21)
        
        k13 = h*f(sun1+0.5*k12,sun2+0.5*k22,sun3+0.5*k32)
        k23 = h*f(sun2+0.5*k22,sun1+0.5*k12,sun3+0.5*k32)
        k33 = h*f(sun3+0.5*k32,sun1+0.5*k12,sun2+0.5*k22)
        
        k14 = h*f(sun1+k13,sun2+k23,sun3+k33)
        k24 = h*f(sun2+k23,sun1+k13,sun3+k33)
        k34 = h*f(sun3+k33,sun1+k13,sun2+k23)
        
        sun1 += (k11+2*k12+2*k13+k14)/6
        sun2 += (k21+2*k22+2*k23+k24)/6
        sun3 += (k31+2*k32+2*k33+k34)/6
        x1.append(sun1[0]),y1.append(sun1[1]),z1.append(sun1[2])
        x2.append(sun2[0]),y2.append(sun2[1]),z2.append(sun2[2])
        x3.append(sun3[0]),y3.append(sun3[1]),z3.append(sun3[2])
        t+=h
    return [x1,y1,z1],[x2,y2,z2],[x3,y3,z3],t
t1 = time.time()
#initial condition 1: they go along x,y,z with the same speed
sun1_0 = np.array([1e11,0,0,50000,0,0],float)
sun2_0 = np.array([0,1e11,0,0,50000,0],float)
sun3_0 = np.array([0,0,1e11,0,0,50000],float)
r1,r2,r3,t = star_pos_RK(sun1_0,sun2_0,sun3_0)
print("For condition 1:")
print("The first sun is ",sun1_0,"\n","The second sun is ",sun2_0,"\n","The third sun is ",sun3_0,sep="")
print("Day :",t/86400)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.plot3D(r1[0], r1[1], r1[2],label = "sun 1", color ="r")
ax.plot3D(r2[0], r2[1], r2[2],label = "sun 2", color ="g")
ax.plot3D(r3[0], r3[1], r3[2],label = "sun 3", color ="b")
ax.plot3D(r1[0][0], r1[1][0], r1[2][0],"o", color ="r")
ax.plot3D(r2[0][0], r2[1][0], r2[2][0],"o", color ="g")
ax.plot3D(r3[0][0], r3[1][0], r3[2][0],"o", color ="b")
ax.legend(loc = 1)
print()

#initial conditon 2: different position 
sun1_0 = np.array([1e11,2e11,-3e11,50000,0,0],float)
sun2_0 = np.array([-3e11,1e11,4e11,50000,0,5000],float)
sun3_0 = np.array([-4e11,-5e11,1e11,50000,0,0],float)
r1,r2,r3,t = star_pos_RK(sun1_0,sun2_0,sun3_0)
print("For condition 2:")
print("The first sun is ",sun1_0,"\n","The second sun is ",sun2_0,"\n","The third sun is ",sun3_0,sep="")
print("Day :",t/86400)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.plot3D(r1[0], r1[1], r1[2],label = "sun 1", color ="r")
ax.plot3D(r2[0], r2[1], r2[2],label = "sun 2", color ="g")
ax.plot3D(r3[0], r3[1], r3[2],label = "sun 3", color ="b")
ax.plot3D(r1[0][0], r1[1][0], r1[2][0],"o", color ="r")
ax.plot3D(r2[0][0], r2[1][0], r2[2][0],"o", color ="g")
ax.plot3D(r3[0][0], r3[1][0], r3[2][0],"o", color ="b")
ax.legend(loc = 1)
print()

#initial condition 3: two stars in a symmetric position related to the remaining star
sun1_0 = np.array([1e10,0,0,-30,10000,0],float)
sun2_0 = np.array([0,0,1e11,-30000,0,50000],float)
sun3_0 = np.array([0,0,-1e11,-30000,0,-50000],float)
r1,r2,r3,t = star_pos_RK(sun1_0,sun2_0,sun3_0)
print("For condition 3:")
print("The first sun is ",sun1_0,"\n","The second sun is ",sun2_0,"\n","The third sun is ",sun3_0,sep="")
print("Day :",t/86400)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.plot3D(r1[0], r1[1], r1[2],label = "sun 1", color ="r")
ax.plot3D(r2[0], r2[1], r2[2],label = "sun 2", color ="g")
ax.plot3D(r3[0], r3[1], r3[2],label = "sun 3", color ="b")
ax.plot3D(r1[0][0], r1[1][0], r1[2][0],"o", color ="r")
ax.plot3D(r2[0][0], r2[1][0], r2[2][0],"o", color ="g")
ax.plot3D(r3[0][0], r3[1][0], r3[2][0],"o", color ="b")
ax.legend(loc = 1)

print(time.time()-t1)
