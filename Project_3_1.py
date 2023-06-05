import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

#The function for iteration
@jit(nopython=True)
def f(r):
    G = 6.6738e-11
    Msun = 1.9891e30
    fx = r[2]
    fy = r[3]
    fvx = -G*Msun*r[0]/(r[0]**2+r[1]**2)**(3/2)
    fvy = -G*Msun*r[1]/(r[0]**2+r[1]**2)**(3/2)
    return np.array([fx,fy,fvx,fvy])
    
@jit(nopython=True)
def star_pos_RK(r0):
    r = r0
    x,y = [r0[0]],[r0[1]]
    t = 0
    h = 450 #7.5mins
    i = 1
    j = 0
    while i > 0:
        k1 = h*f(r)
        k2 = h*f(r+0.5*k1)
        k3 = h*f(r+0.5*k2)
        k4 = h*f(r+k3)
        r = r + (k1+2*k2+2*k3+k4)/6 
        x.append(r[0]),y.append(r[1])
        j+=1
        t+=h
        #use to determine when is the planet finished one cycle and end the loop
        if y[j-1]<0 and y[j]>0:
            i = -1
    return np.array([x,y]),t


#star at y = 0
r0_mercury = np.array([4.6e10,0.0,0.0,57200])
r0_earth = np.array([1.471e11,0.0,0.0,30300])
r0_mars = np.array([2.067e11,0.0,0.0,26400])

t = time.time()
r_mercury, p_mercury = star_pos_RK(r0_mercury)
r_earth, p_earth = star_pos_RK(r0_earth)
r_mars, p_mars = star_pos_RK(r0_mars)

print("time needed = ", time.time() - t,"s")
print("period of mercury", p_mercury/86400, "days")
print("error =", abs(88-p_mercury/86400)/88*100,"%")
print("period of earth",p_earth/86400, "days")
print("error =", abs(365.2-p_earth/86400)/365.2*100,"%")
print("period of mars",p_mars/86400, "days")
print("error =", abs(687-p_mars/86400)/687*100,"%")


plt.plot(r_mercury[0],r_mercury[1], label = "Mercury")
plt.plot(r_earth[0],r_earth[1], label = "Earth")
plt.plot(r_mars[0],r_mars[1], label = "Mars")
plt.plot([0],[0],"o", color = "red", label = "Sun")  
plt.xlabel("x (m)")   
plt.ylabel("y (m)")
plt.title("The Orbital of Mercury, Earth and Mars")
plt.legend(loc = 1)
plt.show()


