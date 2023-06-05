import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit


@jit(nopython=True)
def f(r):
    G = 6.6738e-11
    M_j = 1.898e27
    fx = r[2]
    fy = r[3]
    fvx = -G*M_j*r[0]/(r[0]**2+r[1]**2)**(3/2)
    fvy = -G*M_j*r[1]/(r[0]**2+r[1]**2)**(3/2)
    return np.array([fx,fy,fvx,fvy])

@jit(nopython=True)
def RK(r0,t_needed):
    r = r0
    x,y = [r0[0]],[r0[1]]
    t = 0
    t_list = [0]
    h = 1 #1s
    j = 0
    period = []
    if t_needed == 0: # use to find out the period of ganymede
        i = 1
        while i > 0:
            k1 = h*f(r)
            k2 = h*f(r+0.5*k1)
            k3 = h*f(r+0.5*k2)
            k4 = h*f(r+k3)
            r = r + (k1+2*k2+2*k3+k4)/6 
            x.append(r[0]),y.append(r[1])
            j+=1
            t+=h
            t_list.append(t)
            #use to determine when is the planet finished one cycle and end the loop
            if y[j-1]<0 and y[j]>0:
                i = -1
        return np.array([x,y]),t_list, t_list[-1]
    else: #use the period of ganymede
        while t < t_needed:
            k1 = h*f(r)
            k2 = h*f(r+0.5*k1)
            k3 = h*f(r+0.5*k2)
            k4 = h*f(r+k3)
            r = r + (k1+2*k2+2*k3+k4)/6 
            x.append(r[0]),y.append(r[1])
            j+=1
            t+=h
            t_list.append(t)
            if y[j-1]<0 and y[j]>0:
                period.append(t)
        return np.array([x,y]),t_list, period[0]
    
#initial conditons
Ganymede_0 = np.array([1.069e9,0,0,10890], float)
Europa_0 = np.array([6.648e8,0,0,13870], float)
Io_0 = np.array([4.201e8,0,0,17400], float)

#result of data
Ganymede_pos, Ganymede_t, Ganymede_p = RK(Ganymede_0, 0)
Europa_pos, Europa_t, Europa_p = RK(Europa_0, Ganymede_p)
Io_pos, Io_t, Io_p = RK(Io_0, Ganymede_p)
print("The orbital period of Ganymede is", Ganymede_p/86400, "days")
print("error =",abs(Ganymede_p/86400-7.155)/7.155*100,"%")
print("The orbital period of Europa is", Europa_p/86400, "days")
print("error =",abs(Europa_p/86400-3.551)/3.551*100,"%")
print("The orbital period of Io is", Io_p/86400, "days")
print("error =",abs(Io_p/86400-1.769)/1.769*100,"%")

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
with writer.saving(fig,"Animation_without.mp4",round(len(Ganymede_t)/1024)):
    for i in range(0,len(Ganymede_t),1024): 
        line1.set_data([Ganymede_pos[0,i]],[Ganymede_pos[1,i]])
        line2.set_data([Europa_pos[0,i]],[Europa_pos[1,i]])
        line3.set_data([Io_pos[0,i]],[Io_pos[1,i]]) 
        writer.grab_frame()


