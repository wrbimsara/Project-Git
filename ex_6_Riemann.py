# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:51:05 2026

@author: W R BIMSARA
"""

import numpy as np
import matplotlib.pyplot as pl


#................................................................

#Grid define

#.................................................................


N = 100
dx = 1/N
x = np.linspace (0, 1, N)


#..................................................................

#Empty grids for solution

#..................................................................


u   = np.zeros(N)
rho = np.zeros(N)
e   = np.zeros(N)
p   = np.zeros(N)


#..................................................................

#Consants

#..................................................................


gamma = 1.4
CFL   = 0.8


#..................................................................

#Initial conditions

#..................................................................


for i in range (0,len(x)):
    if x[i] < 0.5 :
        rho[i] = 1. 
        p[i] = 1.
    else :
        rho[i] = 0.125 
        p[i] = 0.1
    
#energy and velocity (initial conditions)
E = p / (gamma - 1) + 0.5 * rho * u**2
u[:] = 0.

U = np.array([rho, rho * u, E])


#...................................................................

#Temporal setting

#...................................................................


t_end   = 0.2
t_start = 0.


#...................................................................

#Define the dictonaries

#...................................................................

#Defin the primitives
def primitives(U):
    rho = U[0]          #density
    u   = U[1] / rho    #velociy
    p   = (gamma -1.) * (U[2] - 0.5 * rho * u**2)  #pressure
    
    return rho, u, p


#Define the flux matrix 
def flux(U):
    rho, u, p = primitives(U)
    
    return np.array([rho * u, rho * (u**2) + p, u * U[2] + u * p])
    

#Define roe-solver
def roeSolver(UL,  UR):
    rhoL, uL, pL = primitives(UL)   #sates left
    rhoR, uR, pR = primitives(UR)   #sataes right
    
    #total enthalpy , H = (e + p) / rho
    HL = (UL[2] + pL) / rhoL
    HR = (UR[2] + pR) / rhoR
    
    sqrhoL = np.sqrt(rhoL)
    sqrhoR = np.sqrt(rhoR)
    
    #roe-averaged qnty
    u = (sqrhoL * uL + sqrhoR * uR) / (sqrhoL + sqrhoR)
    H = (sqrhoL * HL + sqrhoR * HR) / (sqrhoL + sqrhoR)
    a = np.sqrt ( (gamma -1 ) * (H - 0.5 * u**2))
    
    eigenV = np.array([u-a , u, u +a]) # eigen values
    
    #eigen vectors
    v1 = np.array([1, u - a, H - u * a])
    v2 = np.array([1, u, 0.5 * u**2])
    v3 = np.array([1, u + a, H + u * a])
   
    #delta(U) = UR - UL
    dU1 = rhoR - rhoL                 #delta(u)
    dU2 = rhoR * uR - rhoL * uL       #delta(rho*u)
    dU3 = UR[2] - UL[2]               #delta(e)

    #from wave decmpotition (delta(U) = sigma (a i * v i))
    a2 = (gamma - 1) / a**2 * ((H - u**2) * dU1 + u * dU2 - dU3)
    a1 = 0.5 * ((dU1 - a2) - (dU2 - u * dU1) / a)
    a3 = 0.5 * ((dU1 - a2) + (dU2 - u * dU1) / a) 
   
    
   #flux dissipation (D = |lamda i|* a i * v i )
    dissipation = (abs(eigenV[0]) * a1 * v1 +
                   abs(eigenV[1]) * a2 * v2 +
                   abs(eigenV[2]) * a3 * v3)
    
    return 0.5 * (flux(UL) + flux(UR)) - 0.5 * dissipation #roe flux/numerial flux
   

def godunov(U, dt, dx):
    
    F = np.zeros((3, N - 1))
    for i in range(N - 1):
        F[:, i] = roeSolver(U[:, i], U[:, i + 1])

    U_new = U.copy()
    U_new[:, 1:-1] -= dt / dx * (F[:, 1:] - F[:, :-1])
    U_new[:, 0] = U_new[:, 1]      #BCs
    U_new[:, -1] = U_new[:, -2]
    return U_new

    
#...................................................................

# time loop

#...................................................................

   
while t_start < t_end:
        rho, u, p = primitives(U)
        a = np.sqrt(gamma * p / rho)
        dt = min(CFL * dx / np.max(np.abs(u) + a), t_end - t_start)
        U = godunov(U, dt, dx)
        t_start += dt    
   
rho, u, p = primitives(U)
e = p / ((gamma - 1) * rho)    


#....................................................................
        
#Graph plot 

#....................................................................
   
pl.plot(x,rho,'b',label='Density')
pl.plot(x,u,'g',label='Velocity')
pl.plot(x,p,'r',label='Pressure')
pl.plot(x,e,'k',label='Energy')
#pl.plot(x, rho28, 'r', label="Numerical (t = 0.288")
pl.xlabel("x")
pl.ylabel("Density, velocity, energy, pressure(x,t)")
pl.legend()
pl.grid()
pl.xlim([0.,1.])
pl.ylim([-0.2,3.1])
pl.show()
   
    
   
    
   
    