# -*- coding: 

"""
LSNN project
@author: faramarz.faghihi
One input layer connected to two E layers one of them Adaptive
 and one I layer
"""
import matplotlib.pyplot as plt
import numpy as np           # NumPy is the fundamental package for scientific computing 
import random                # Generate pseudo-random numbers
import tensorflow as tf
import numpy.random as rd
import time
import pickle
from tensorflow.examples.tutorials.mnist import input_data
plt.close('all')


mnist = input_data.read_data_sets("../datasets/MNIST", one_hot=True)

##################### Define the main hyper parameter accessible from the shell ############################################


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('n_epochs', 10, 'number of iteration (55000 is one epoch)')
tf.app.flags.DEFINE_integer('batch', 10, 'number of iteration (55000 is one epoch)')
tf.app.flags.DEFINE_integer('print_every', 100, 'print every k steps')
tf.app.flags.DEFINE_integer('n1', 30, 'Number of neurons in the first hidden layer')
tf.app.flags.DEFINE_integer('n2', 30, 'Number of neurons in the second hidden layer')
#
tf.app.flags.DEFINE_float('p01', .001, 'Proportion of connected synpases at initialization')
tf.app.flags.DEFINE_float('p02', .003, 'Proportion of connected synpases at initialization')
tf.app.flags.DEFINE_float('p0out', .3, 'Proportion of connected synpases at initialization')
tf.app.flags.DEFINE_float('l1', 1e-5, 'l1 regularization coefficient')
tf.app.flags.DEFINE_float('gdnoise', 1e-5, 'gradient noise coefficient')
tf.app.flags.DEFINE_float('lr', 0.5, 'Learning rate')


##################### Define useful constants ###############################################################################


dtype = tf.float32
n_pixels = 28 * 28
n_1 = FLAGS.n1
n_2 = FLAGS.n2
n_out = 10
n_image_per_epoch = mnist.train.images.shape[0]
n_iter = FLAGS.n_epochs * n_image_per_epoch // FLAGS.batch

print_every = FLAGS.print_every
n_minibatch = FLAGS.batch
lr = FLAGS.lr


#################### Define the number of neurons per layer#################################################################


sparsity_list = [FLAGS.p01, FLAGS.p02, FLAGS.p0out]
nb_non_zero_coeff_list = [n_pixels * n_1 * FLAGS.p01, n_1 * n_2 * FLAGS.p02, n_2 * n_out * FLAGS.p0out]
nb_non_zero_coeff_list = [int(n) for n in nb_non_zero_coeff_list]



##########  Set up some parameters for the simulation

T         = 10    # total time to sumulate (msec)
dt        = 0.01 # Simulation timestep
time      = int(T / dt)
V_spike   = 0.5
inpt      = 1.0   # Neuron input voltage
neuron_input=np.full((time),inpt)

################    SNN structure  ###################
num_layers_Input = 1
num_layers_Exc   = 1
num_layers_Adap  = 1
num_layers_Inh   = 1
########

num_neurons_Input = 100
num_neurons_Exc   = 10
num_neurons_Adap  = 4
num_neurons_Inh   = 10

########### Functions to graph the results ##############################

def plot_neuron_behaviour(time, data, neuron_type, neuron_id, y_title):
    plt.plot(time,data)
    plt.title('{} @ {}'.format(neuron_type, neuron_id))
    plt.ylabel(y_title)
    plt.xlabel('Time (msec)')
    # Graph to the data with some headroom...
    y_min = 0
    y_max = max(data)*1.2
    if y_max == 0:
        y_max = 1
    plt.ylim([y_min,y_max])   
    plt.show()
    
    
def plot_membrane_potential(time, Vm, neuron_type, neuron_id = 0):
    plot_neuron_behaviour(time, Vm, neuron_type, neuron_id, y_title = 'Membrane potential (V)')

def plot_spikes(time, Vm, neuron_type, neuron_id = 0):
    plot_neuron_behaviour(time, Vm, neuron_type, neuron_id, y_title = 'Spike (V)')
    
 #################   Integrate and Fire neuron model for Input layer neurons ##################
    
class LIFNeuron():
     def __init__(self, debug=True):
        
        self.dt       = 0.125       # simulation time step
        self.t_rest   = 0           # initial refractory time
        
        #LIF Properties 
        self.Vm       = np.array([0])    # Neuron potential (mV)
        self.Vth       = np.array([0])    # threshold (mV)
        self.time     = np.array([0])    # Time duration for the neuron (needed?)
        self.spikes   = np.array([0])    # Output (spikes) for the neuron
        
        self.t        = 0                # Neuron time step
        self.Rm       = 1                # Resistance (kOhm)
        self.Cm       = 10               # Capacitance (uF) 
        self.tau_m    = self.Rm * self.Cm # Time constant
        self.tau_ref  = 4                # refractory period (ms)
        self.Vth      = 0.3             # = 1  #spike threshold
        self.V_spike  = 1                # spike delta (V)
        self.type     = 'Leaky Integrate and Fire'
        self.debug    = debug
        if self.debug:
            print ('LIFNeuron(): Created {} neuron starting at time {}'.format(self.type, self.t))
    
     def spike_generator(self, neuron_input):
        # Create local arrays for this run
        duration = len(neuron_input)
        Vm = np.zeros(duration)  #len(time)) # potential (V) trace over time
        time = np.arange(self.t, self.t+duration)       
        spikes = np.zeros(duration)  #len(time))
        
        if self.debug:
            print ('spike_generator(): Running time period self.t={}, self.t+duration={}'
                   .format(self.t, self.t+duration))
        
        #######    Seed the new array with previous value of last run
        Vm[-1] = self.Vm[-1]
               
        if self.debug:
            print ('LIFNeuron.spike_generator.initial_state(input={}, duration={}, initial Vm={}, t={})'
               .format(neuron_input, duration, Vm[-1], self.t))
            
        for i in range(duration):
            if self.debug:
                print ('Index {}'.format(i))
                
            if self.t > self.t_rest:
                Vm[i]=Vm[i-1] + (-Vm[i-1] + neuron_input[i-1]*self.Rm) / self.tau_m * self.dt
               
            if self.debug:
                    print('spike_generator(): i={}, self.t={}, Vm[i]={}, neuron_input={}, self.Rm={}, self.tau_m * self.dt = {}'
                          .format(i,self.t, Vm[i], neuron_input[i], self.Rm, self.tau_m * self.dt))
                
            if Vm[i] >= self.Vth:
                    Vm[i] += V_spike
                    spikes[i] += self.V_spike
                    self.t_rest = self.t + self.tau_ref
     #  This is the place for adaptation # self.Vth = self.Vth +0.4
                    if self.debug:
                        print ('*** LIFNeuron.spike_generator.spike=(self.t_rest={}, self.t={}, self.tau_ref={})'
                           .format(self.t_rest, self.t, self.tau_ref))

            self.t += self.dt
        
        # Save state
        self.Vm = np.append(self.Vm, Vm)
        self.spikes = np.append(self.spikes, spikes)
        self.time = np.append(self.time, time)
        
        if self.debug:
            print ('LIFNeuron.spike_generator.exit_state(Vm={} at iteration i={}, time={})'
                   .format(self.Vm, i, self.t))
        

############## Function to create a specified number of layers with specified number of neurons 
    
def create_neurons_Input(num_layers_Input, num_neurons_Input, debug=True):
     neurons_Input = []
     for layer in range(num_layers_Input):
        if debug:
            print ('create_neurons(): Creating layer {}'.format(layer))
        neuron_layer_Input = []
        for count in range(num_neurons_Input):
            neuron_layer_Input.append(LIFNeuron(debug=debug))
        neurons_Input.append(neuron_layer_Input)
     return neurons_Input
#####################
     
neurons_Input = create_neurons_Input(num_layers_Input, num_neurons_Input, debug=False) 

##########   Integrate and Fire neuron model for excitatory neurons ##################
class LIFNeuron_Exc():
     def __init__(self, debug=True):
        
        self.dt       = 0.125       # simulation time step
        self.t_rest   = 0           # initial refractory time
        
        #LIF Properties 
        self.Vm       = np.array([0])    # Neuron potential (mV)
        self.Vth       = np.array([0])    # threshold (mV)
        self.time     = np.array([0])    # Time duration for the neuron (needed?)
        self.spikes   = np.array([0])    # Output (spikes) for the neuron
        
        self.t        = 0                # Neuron time step
        self.Rm       = 1                # Resistance (kOhm)
        self.Cm       = 10               # Capacitance (uF) 
        self.tau_m    = self.Rm * self.Cm # Time constant
        self.tau_ref  = 4                # refractory period (ms)
        self.Vth      = 0.05             # = 1  #spike threshold
        self.V_spike  = 1                # spike delta (V)
        self.type     = 'Leaky Integrate and Fire'
        self.debug    = debug
        if self.debug:
            print ('LIFNeuron_Inh(): Created {} neuron starting at time {}'.format(self.type, self.t))
    
     def spike_generator_Exc(self, layer_spikes):
        # Create local arrays for this run
        duration = len(layer_spikes)
        Vm = np.zeros(duration)  #len(time)) # potential (V) trace over time
        time = np.arange(self.t, self.t+duration)       
        spikes = np.zeros(duration)  #len(time))
        
        if self.debug:
            print ('spike_generator(): Running time period self.t={}, self.t+duration={}'
                   .format(self.t, self.t+duration))
        
        #######    Seed the new array with previous value of last run
        Vm[-1] = self.Vm[-1]
               
        if self.debug:
            print ('LIFNeuron_Inh.spike_generator.initial_state(input={}, duration={}, initial Vm={}, t={})'
               .format(layer_spikes, duration, Vm[-1], self.t))
            
        for i in range(duration):
            if self.debug:
                print ('Index {}'.format(i))
                
            if self.t > self.t_rest:
                Vm[i]=Vm[i-1] + (-Vm[i-1] + layer_spikes[i-1]*self.Rm) / self.tau_m * self.dt
               
            if self.debug:
                    print('spike_generator(): i={}, self.t={}, Vm[i]={}, neuron_input={}, self.Rm={}, self.tau_m * self.dt = {}'
                          .format(i,self.t, Vm[i], layer_spikes[i], self.Rm, self.tau_m * self.dt))
                
            if Vm[i] >= self.Vth:
                    Vm[i] += V_spike
                    spikes[i] += self.V_spike
                    self.t_rest = self.t + self.tau_ref

                    if self.debug:
                        print ('*** LIFNeuron_Inh.spike_generator.spike=(self.t_rest={}, self.t={}, self.tau_ref={})'
                           .format(self.t_rest, self.t, self.tau_ref))

            self.t += self.dt
        
        # Save state
        self.Vm = np.append(self.Vm, Vm)
        self.spikes = np.append(self.spikes, spikes)
        self.time = np.append(self.time, time)
        
        if self.debug:
            print ('LIFNeuron_Inh.spike_generator.exit_state(Vm={} at iteration i={}, time={})'
                   .format(self.Vm, i, self.t))
        
        #return time, Vm, output             
############## Function to create a specified number of layers with specified number of neurons 
def create_neurons_Exc(num_layers_Exc, num_neurons_Exc, debug=True):
        neurons_Exc = []
        neuron_layer_Exc = []
        for count in range(num_neurons_Exc):
            neuron_layer_Exc.append(LIFNeuron_Exc(debug=debug))
        neurons_Exc.append(neuron_layer_Exc)
        return neurons_Exc     
 

neurons_Exc = create_neurons_Exc(num_layers_Exc, num_neurons_Exc, debug=False)  
#######################################################################

#######   Integrate and Fire neuron model for Inhibitory neurons ##################
    
class LIFNeuron_Inh():
     def __init__(self, debug=True):
        
        self.dt       = 0.125       # simulation time step
        self.t_rest   = 0           # initial refractory time
        
        #LIF Properties 
        self.Vm       = np.array([0])    # Neuron potential (mV)
        self.Vth       = np.array([0])    # threshold (mV)
        self.time     = np.array([0])    # Time duration for the neuron (needed?)
        self.spikes   = np.array([0])    # Output (spikes) for the neuron
        
        self.t        = 0                # Neuron time step
        self.Rm       = 1                # Resistance (kOhm)
        self.Cm       = 10               # Capacitance (uF) 
        self.tau_m    = self.Rm * self.Cm # Time constant
        self.tau_ref  = 4                # refractory period (ms)
        self.Vth      = 0.2             # = 1  #spike threshold
        self.V_spike  = 1                # spike delta (V)
        self.type     = 'Leaky Integrate and Fire'
        self.debug    = debug
        if self.debug:
            print ('LIFNeuron_Inh(): Created {} neuron starting at time {}'.format(self.type, self.t))
    
     def spike_generator_Inh(self, neuron_input):
        # Create local arrays for this run
        duration = len(neuron_input)
        Vm = np.zeros(duration)  #len(time)) # potential (V) trace over time
       
        time = np.arange(self.t, self.t+duration)       
        spikes = np.zeros(duration)  #len(time))
        
        if self.debug:
            print ('spike_generator(): Running time period self.t={}, self.t+duration={}'
                   .format(self.t, self.t+duration))
        
        #######    Seed the new array with previous value of last run
        Vm[-1] = self.Vm[-1]
               
        if self.debug:
            print ('LIFNeuron_Inh.spike_generator.initial_state(input={}, duration={}, initial Vm={}, t={})'
               .format(neuron_input, duration, Vm[-1], self.t))
            
        for i in range(duration):
            if self.debug:
                print ('Index {}'.format(i))
                
            if self.t > self.t_rest:
                Vm[i]=Vm[i-1] + (-Vm[i-1] + neuron_input[i-1]*self.Rm) / self.tau_m * self.dt
               
            if self.debug:
                    print('spike_generator(): i={}, self.t={}, Vm[i]={}, neuron_input={}, self.Rm={}, self.tau_m * self.dt = {}'
                          .format(i,self.t, Vm[i], neuron_input[i], self.Rm, self.tau_m * self.dt))
                
            if Vm[i] >= self.Vth:
                    Vm[i] += V_spike
                    spikes[i] += self.V_spike
                    self.t_rest = self.t + self.tau_ref
     #  This is the place for adaptation # self.Vth = self.Vth +0.4
                    if self.debug:
                        print ('*** LIFNeuron_Inh.spike_generator.spike=(self.t_rest={}, self.t={}, self.tau_ref={})'
                           .format(self.t_rest, self.t, self.tau_ref))

            self.t += self.dt
        
        # Save state
        self.Vm = np.append(self.Vm, Vm)
        self.spikes = np.append(self.spikes, spikes)
        self.time = np.append(self.time, time)
        
        if self.debug:
            print ('LIFNeuron_Inh.spike_generator.exit_state(Vm={} at iteration i={}, time={})'
                   .format(self.Vm, i, self.t))
        
############## Function to create a specified number of layers with specified number of neurons 
    
def create_neurons_Inh(num_layers_Inh, num_neurons_Inh, debug=True):
     neurons_Inh = []
     for layer in range(num_layers_Inh):
        if debug:
            print ('create_neurons(): Creating layer {}'.format(layer))
        neuron_layer_Inh = []
        for count in range(num_neurons_Inh):
            neuron_layer_Inh.append(LIFNeuron_Inh(debug=debug))
        neurons_Inh.append(neuron_layer_Inh)
     return neurons_Inh  

neurons_Inh = create_neurons_Inh(num_layers_Inh, num_neurons_Inh, debug=False)  

######################################################################  
############# Run stimuli for each neurons in layer input ##################

stimulus_len = len(neuron_input)
layer = 0
for neuron in range(num_neurons_Input):
       offset = random.randint(0,100)   # Simulates stimulus starting at different times
       stimulus = np.zeros_like(neuron_input)
       stimulus[offset:stimulus_len] = neuron_input[0:stimulus_len - offset]
       neurons_Input[layer][neuron].spike_generator(stimulus)

    
layer = 0 
layer_spikes = np.zeros_like(neurons_Input[layer][0].spikes)
for i in range(num_neurons_Input):
     layer_spikes += neurons_Input[layer][i].spikes 
       
    
   ###### Now simulate spikes propogated to a neuron on Excitatory layer
for i in range(0 , num_neurons_Exc):  
 neurons_Exc[0][i].spike_generator_Exc(layer_spikes)

##### Now simulate spikes propogated to a neuron on Excitatory layer
 for i in range(0 , num_neurons_Inh):  
  neurons_Inh[0][i].spike_generator_Inh(layer_spikes)