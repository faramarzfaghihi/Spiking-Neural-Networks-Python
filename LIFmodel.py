# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 14:14:50 2019 by Faramarz Faghihi

@author: Novin Pendar
"""

from numpy import *
from pylab import *
## setup parameters and state variables

T = 150 # total time to simulate (msec)
dt = 0.125 # simulation time step (msec)
time = arange(0, T+dt, dt) # time array
t_rest = 0 # initial refractory time

## LIF properties
Vm = zeros(len(time)) # potential (V) trace over time
teta = zeros(len(time))+1 # threshold initialization
Rm = 1 # resistance (kOhm)
Cm = 10 # capacitance (uF)
tau_m = Rm*Cm # time constant (msec)
tau_ref = 4 # refractory period (msec)
Vth = 1 # spike threshold (V)
V_spike = 0.5 # spike delta (V)

## Input stimulus
I = 1.5 # input current (A)
ro = 1 # increase step after one spike
tau_d = 450 # contant decay 
## iterate over each time step

for i, t in enumerate(time):
#   teta [i] = teta[i-1] - (1/tau_d)*(teta [i-1]);
   if t > t_rest:
     Vm[i] = Vm[i-1] + (-Vm[i-1] + I*Rm) / tau_m * dt
  
#   if Vm[i] >= teta[i]:
   if Vm[i] >= Vth:   
     Vm[i] += V_spike
     t_rest = t + tau_ref
    # teta [i] = teta[i-1] - (1/tau_d)*(teta [i-1])+ ro;
#     teta [i] +=  ro;
      
   
   
## plot membrane potential trace
   
plot(time, Vm)
title('Leaky Integrate-and-Fire Example, by Faramarz Faghihi')
ylabel('Membrane Potential (V)')
xlabel('Time (msec)')
ylim([0,2])
show()

plot(time, teta)
title('threshold dynamics, by Faramarz Faghihi')
ylabel('Threshold')
xlabel('Time (msec)')
ylim([0,10])
show()


