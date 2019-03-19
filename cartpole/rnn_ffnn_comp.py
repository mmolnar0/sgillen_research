# Force keras to use the CPU because it's actually faster for this size network
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' 
os.environ["CUDA_VISIBLE_DEVICES"] = ''
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad

import numpy as np
import matplotlib.pyplot as plt
from math import pi

from IPython import get_ipython
ipython = get_ipython()
ipython.magic('matplotlib qt')

from cartpole_class import Cartpole 
import nn_tools

     
#Close all windows
plt.close('all')

## Define some constants

# time vector, doesn't actually affect the integration, only what times it records our state variable at
dt = 0.01
endTime = 13
time = np.arange(0.0, endTime, dt)
numSamples = int(endTime/dt)
u_max = 1000          #max value for control input
Ts = 1                #Sampling rate of expert trajectory
look_back = 3         #Take look_back-1 samples from the past
swingup_lim = 550     #Where to slice expert trajectory in swingup and balance part
num_trials = 40       #Number of Training sets from expert controller

# Cartpole is a class we defined that takes care of the simulation/animation of the cartpole
bot = Cartpole(time, u_max, Ts, look_back)
#Get trajectories for num_trials simulations and one standard case trajectory
[y, u, y_std, u_std, ind_bal] = bot.simulate_ES_control(num_trials)
#Get number of samples summed over all training sets for balance and swingup
samples_bal_tot = sum(y.shape[0]-np.array(ind_bal))
samples_swingup_tot = numSamples*num_trials - samples_bal_tot
#Sample the Data for the standard case
y_lb_std =  nn_tools.sampleData(y_std, look_back, Ts)

#Trials concatenated
y_lb_m = np.zeros((y.shape[0]*y.shape[2],y.shape[1],look_back)) #All trials states with look_back concatenated
y_m = np.zeros((y.shape[0]*y.shape[2],y.shape[1]))              #All trials states without look_back concatenated
u_m = np.zeros((y.shape[0]*y.shape[2],1))                       #All trials control inputs concatenated
y_m_swingup = np.zeros((samples_swingup_tot,y.shape[1]))        #All trials states concatenated only swingup
u_m_swingup = np.zeros((samples_swingup_tot,1))                 #All trials control inputs concatenated only swingup
y_m_bal = np.zeros((samples_bal_tot, y.shape[1]))               #All trials states concatenated only balance
u_m_bal = np.zeros((samples_bal_tot,1))                         #All trials control inputs concatenated only balance
switch_log = np.zeros((numSamples*num_trials,2))                #Logic array, deciding between two parts of the neural network
ind_bal = np.array(ind_bal)                                     #Index for every trial marking where expert controller switches over to LQR

#Fill the initialized matrices with values according to their purpose described above
for trial in range(y.shape[-1]):
    y_lb_m[trial*numSamples:(trial+1)*numSamples,:,:] = nn_tools.sampleData(y[:,:,trial], look_back, Ts)
    y_m[trial*numSamples:(trial+1)*numSamples,:] = y[:,:,trial]
    u_m[trial*numSamples:(trial+1)*numSamples,0:1] = u[:,0:1,trial]
    switch_log[trial*numSamples:trial*numSamples+ind_bal[trial],:] = np.array([0,1])
    switch_log[trial*numSamples+ind_bal[trial]:(trial+1)*numSamples,:] = np.array([1,0])
    
    up_swingup = sum(ind_bal[:trial])+ind_bal[trial]
    low_swingup = sum(ind_bal[:trial])
    if(trial == 0):
        low_swingup = 0
    up_bal =sum(y.shape[0]-ind_bal[:trial])+y.shape[0]-ind_bal[trial]
    low_bal = sum(y.shape[0]-ind_bal[:trial])
    if(trial == 0):
        low_bal = 0 
    y_m_swingup[low_swingup:up_swingup,:] = y[:ind_bal[trial],:,trial]
    u_m_swingup[low_swingup:up_swingup,0:1] = u[:ind_bal[trial],0:1,trial]    
    y_m_bal[low_bal:up_bal,:] = y[ind_bal[trial]:,:,trial]
    u_m_bal[low_bal:up_bal,0:1] = u[ind_bal[trial]:,0:1,trial]

#Generate datasets with just the balancing part
NSets = 3  
tSet = 2.5 #Simulation seconds of one set
NSamples = int(tSet/bot.dt)
[y_bal, u_bal] = bot.genStabData(NSets, NSamples)

#Noise which might improve training (just for the one standard trial)
sigma = 0   #variance
noise    = np.random.randn(y_std.shape[0],y_std.shape[1]) * sigma
noise_lb = np.random.randn(y_lb_std.shape[0],y_lb_std.shape[1],y_lb_std.shape[2]) * sigma
noise_bal = np.random.randn(y_bal.shape[0],y_bal.shape[1],y_bal.shape[2]) * sigma
 
#Standard trial dataset sliced in swingup and balance with/without LB
y_lb_swingup = y_lb_std[0:swingup_lim,:,:]
u_lb_swingup = u_std[0:swingup_lim]
noise_lb_swingup = noise_lb[0:swingup_lim,:,:]
y_lb_stabilize = y_lb_std[swingup_lim:,:,:]
u_lb_stabilize = u_std[swingup_lim:]
noise_lb_stabilize = noise_lb[swingup_lim:,:,:]

y_swingup = y_std[0:swingup_lim,:]
u_swingup = u_std[0:swingup_lim,]
y_stabilize = y_std[swingup_lim:,:]
u_stabilize = u_std[swingup_lim:]

#Combined dataset of swingup part and generated balance part (standard trial swingup mixed with several short runs of just balancing)
y_tot = np.concatenate((y_lb_swingup,y_bal), axis = 0)
u_tot = np.concatenate((u_lb_swingup,u_bal), axis = 0)  

#Generate data mesh for learning directly from expert mapping from y to u
# Generate two 1-D arrays: u, v
u = np.linspace(-3, 4, 1000)
v = np.linspace(-7, 7, 1000)

# Generate 2-D arrays from u and v: X, Y
X,Y = np.meshgrid(u, v)
theta = X.flatten()
theta_dot = Y.flatten()
u_map = np.zeros((u.size*v.size,1))
y_map = np.zeros((u.size*v.size,4))
# Compute Z based on X and Y
for i in range(v.size*u.size):    
        y_map[i,:] = np.array([theta[i], 0, theta_dot[i], 0])
        u_map[i] = bot.controlES(y_map[i,:])
        
        

#Print time while simulation run
bot.verbose = 0
#Number of runs of training new networks
networkRuns = 1
#Variable for storing t_err and max(u) 
store = np.zeros((networkRuns,8))
#Loop for several training runs with new networks
for i in range(networkRuns):
    #Set maximum value for u
    bot.u_max = 1000

    ###########################################################################
    """
    Train + Simulate  FF, FFLB and LSTM
    """
#    #Train Feedforward network
#    net_type = 'FF'
#    ff_model = nn_tools.createNet(net_type)
#    ff_history = ff_model.fit(y, u, batch_size = y.shape[0], epochs=5000, verbose=1)
#    #nn_tools.plotTrain(ff_history)
#    [y_ff, u_ff_sim] = bot.simulate_control(ff_model, net_type)
#    print('FF',i, 'loss', ff_history.history['loss'][-1])
#
#    #Train Feedforward network with looking back
#    net_type = 'FFLB'
#    fflb_model = nn_tools.createNet(net_type, look_back)
#    fflb_history = fflb_model.fit(y_lb, u, batch_size = y_lb.shape[0], epochs=5000, verbose=1)
#    #nn_tools.plotTrain(fflb_history)
#    [y_fflb, u_fflb_sim] = bot.simulate_control(fflb_model, net_type)
#    print('FFLB',i, 'loss', fflb_history.history['loss'][-1])
#    
#    #Train LSTM
#    net_type = 'LSTM'
#    lstm_model = nn_tools.createNet(net_type, look_back)
#    lstm_history = lstm_model.fit(np.swapaxes(y_lb,1,2), u, batch_size = y_lb.shape[0], epochs=1000, verbose=1)
#    #nn_tools.plotTrain(lstm_history)
#    [y_lstm, u_lstm_sim] = bot.simulate_control(lstm_model, net_type)
#    print('LSTM',i, 'loss', lstm_history.history['loss'][-1])
    ###########################################################################
    
    ###########################################################################
    """
    Train + Simulate a parallel network without using LQR
    """
    #Turn of LQR
    bot.flag_LQR = 1
    #Names of parallel layers
    layers_Swing = ['n21','n22','n23']
    layers_Bal = ['n11','n12']
    #Net type
    net_type = 'Parallel'
    #Create Network in several versions depending on which inputs/outputs wanted
    parallel_model_ext_log, parallel_model_int_log, parallel_model_inspect = nn_tools.createNet(net_type, look_back)
#    parallel_model, parallel_model_inspect = nn_tools.createNet(net_type, look_back)    
    
    ###########################################################################
    #Switching fixed weights frequently to train alternating with swingup and balance data
    
    #Training runs
#    runs = 200
#    for i in range(runs):
#        #Train Swing-Up part
#        for layer in layers_Swing:
#             parallel_model.get_layer(layer).trainable = True
#        for layer in layers_Bal:
#             parallel_model.get_layer(layer).trainable = False
#        
#        parallel_model.compile(loss='mean_squared_error', optimizer='Adam')
#        parallel_history = parallel_model.fit(y_m_swingup, u_m_swingup, batch_size = y_m_swingup.shape[0],
#                                                          epochs=30, verbose=1)
#        #Train Balancing part
#        for layer in layers_Swing:
#             parallel_model.get_layer(layer).trainable = False
#        for layer in layers_Bal:
#             parallel_model.get_layer(layer).trainable = True
#             
#        parallel_model.compile(loss='mean_squared_error', optimizer='Adam')     
#        parallel_history = parallel_model.fit(y_m_bal, u_m_bal, batch_size = y_m_bal.shape[0],
#                                                         epochs=30, verbose=1)
#        print('--------------------')
#        print(i)
#        print('--------------------')

     ##########################################################################   
     
     ##########################################################################
     #First train with external logic then train with internal logic generation
     #(turn trainability of weights which are not the internal logic off if wanted)

    #Callback objects for tracking data in the training process  
    history_ext_log = nn_tools.InputHistory_lb_extLog()   
    history_int_log = nn_tools.InputHistory_lb() 
    
    #Train network with external logic
    parallel_model_ext_log.compile(loss='mean_squared_error', optimizer=Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
    parallel_history = parallel_model_ext_log.fit([np.swapaxes(y_lb_m,1,2), switch_log], u_m, batch_size = y_m.shape[0],
                                           epochs=240, verbose=1, callbacks=[history_ext_log])  
    parallel_model_ext_log.compile(loss='mean_squared_error', optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
    parallel_history = parallel_model_ext_log.fit([np.swapaxes(y_lb_m,1,2), switch_log], u_m, batch_size = y_m.shape[0],
                                           epochs=40, verbose=1, callbacks=[history_ext_log])  
    
    #BEFORE TRAINING INTERNAL LOGIC MAKE SURE THAT IT ACTUALLY WORKS WITH EXTERNAL LOGIC BEFORE
    #    #Quick test of extern logic network
    theta = 0
    x = 0
    th_dot = 0
    xdot = 0
    init_state = np.array([theta, x, th_dot, xdot])
    [y_ext, dummy] = bot.simulate_control(parallel_model_ext_log, 'Parallel_ext_log', init_state)
    bot.animate_cart(y_ext)
    
    #Switch off trainability of control input generating parts
    for layer in layers_Bal+layers_Swing:
        parallel_model_int_log.get_layer(layer).trainable = False
    
    #Train network with internal logic
    parallel_model_int_log.compile(loss='mean_squared_error', optimizer=Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
    parallel_history = parallel_model_int_log.fit(np.swapaxes(y_lb_m,1,2), u_m, batch_size = y_m.shape[0],
                                           epochs=180, verbose=1, callbacks=[history_int_log])     
    parallel_model_int_log.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
    parallel_history = parallel_model_int_log.fit(np.swapaxes(y_lb_m,1,2), u_m, batch_size = y_m.shape[0],
                                           epochs=200, verbose=1, callbacks=[history_int_log])   
        
    #Quick test of internal logic network (if not working, you might wanna train external logic and internal logic again)
    theta = 0
    x = 0
    th_dot = 0
    xdot = 0    
    init_state = np.array([theta, x, th_dot, xdot])
    [y_int, dummy] = bot.simulate_control(parallel_model_int_log, 'LSTM', init_state)
    bot.animate_cart(y_int)
    
    
    
     ##########################################################################
     
    """
    Train mapping from expert controller with theta and theta_dot
    """
    
    ###########################################################################+
#    history = InputHistory_ff() 
#    #    #Train Feedforward network
#    net_type = 'FF'
#    ff_model = nn_tools.createNet(net_type)
#    ff_history = ff_model.fit(y_map, u_map, batch_size = y_map.shape[0], epochs=5000, verbose=1, callbacks = [history])
#    #nn_tools.plotTrain(ff_history)
#    [y_map_ff, u_map_ff_sim] = bot.simulate_control(ff_model, net_type)
#    print('FF',i, 'loss', ff_history.history['loss'][-1])
     
    
    
    ###########################################################################
    
    
    
    ###########################################################################
 
    


    
    """
    Evaluation for several training runs
    """
    ###########################################################################
#    Y = np.zeros((len(time),4,4,1))
#    Y[:,:,0,0] = y.squeeze()
#    Y[:,:,1,0] = y_ff
#    Y[:,:,2,0] = y_fflb
#    Y[:,:,3,0] = y_lstm
#    LABEL_ROWS = ['ES','FF','FFLB','LSTM']
#    LABEL_COLS = ['CTRL_TYPE']
#    info = [['' for x in range(1)] for y in range(4)]
#    bot.animate_cart_dim(Y, LABEL_ROWS,LABEL_COLS, info)
#
#    print('ES max(u)', np.max(np.abs(u)))
#    print('FF max(u)', np.max(np.abs(u_ff_sim)))
#    print('FFLB max(u)', np.max(np.abs(u_fflb_sim)))
#    print('LSTM max(u)', np.max(np.abs(u_lstm_sim)))
##    print('LSTM parallel max(u)', np.max(np.abs(u_lstm_parallel_sim)))
#
#    print('ES t_err', bot.calc_feat(y.squeeze()))
#    print('FF t_err', bot.calc_feat(y_ff))
#    print('FFLB t_err', bot.calc_feat(y_fflb))
#    print('LSTM t_err', bot.calc_feat(y_lstm))
##   print('LSTM parallel t_err', bot.calc_feat(y_parallel_lstm))
#    
#    store[i,0] = np.max(np.abs(u))
#    store[i,1] = np.max(np.abs(u_ff_sim))
#    store[i,2] = np.max(np.abs(u_fflb_sim))
#    store[i,3] = np.max(np.abs(u_lstm_sim))
#
#    store[i,4] = bot.calc_feat(y.squeeze())
#    store[i,5] = bot.calc_feat(y_ff)
#    store[i,6] = bot.calc_feat(y_fflb)
#    store[i,7] = bot.calc_feat(y_lstm)
#    print(i)
    ###########################################################################
    
"""
Plotting mappings for neural networks and expert controller
"""
#Mapping for external logic network 
X, Y, INPUT, INPUT_EXPERT = nn_tools.getControlMesh_lb_extLog(bot, parallel_model_ext_log, theta_lim = [-pi*1.2, pi*1.2], theta_dot_lim = [-7, 7])
ind_balSwing = np.where((140 * (pi/180)< X[0,:]))[0][0]
# Display the resulting image with pcolor()
INPUT = INPUT[:-1, :-1]

fig = plt.figure()
ax = plt.axes()
c = ax.pcolor(X, Y, INPUT, cmap='RdBu', vmin=INPUT.min(), vmax=INPUT.max())
fig.colorbar(c, ax=ax)
plt.title('External Logic - Control Value')
plt.show()


X, Y, INPUT, _, L1, L2 = nn_tools.getControlMesh_lb_IntLog(bot, parallel_model_int_log, parallel_model_inspect)
# Display the resulting image with pcolor()
L1 = L1[:-1, :-1]
fig = plt.figure()
ax = plt.axes()
c = ax.pcolor(X, Y, L1, cmap='RdBu', vmin=L1.min(), vmax=L1.max())
fig.colorbar(c, ax=ax)
plt.title('Internal Logic - L1')
plt.show()

# Display the resulting image with pcolor()
L2 = L2[:-1, :-1]
fig = plt.figure()
ax = plt.axes()
c = ax.pcolor(X, Y, L2, cmap='RdBu', vmin=L2.min(), vmax=L2.max())
fig.colorbar(c, ax=ax)
plt.title('Internal Logic - L2')
plt.show()

# Display the resulting image with pcolor()
INPUT = INPUT[:-1, :-1]

fig = plt.figure()
ax = plt.axes()
c = ax.pcolor(X, Y, INPUT, cmap='RdBu', vmin=INPUT.min(), vmax=INPUT.max())
fig.colorbar(c, ax=ax)
plt.title('Internal Logic - Control Value')
plt.show()

# Display the resulting image with pcolor()
INPUT_EXPERT_sliced = INPUT_EXPERT[:-1, :-1]

fig = plt.figure()
ax = plt.axes()
c = ax.pcolor(X, Y, INPUT_EXPERT_sliced, cmap='RdBu', vmin=INPUT_EXPERT.min(), vmax=INPUT_EXPERT.max())
fig.colorbar(c, ax=ax)

plt.title('Expert - Control Value')
plt.show()


# Display the resulting image with pcolor()
INPUT_EXPERT_sliced = INPUT_EXPERT[:-1, :-1]

fig = plt.figure()
ax = plt.axes()
c = ax.pcolor(X, Y, INPUT_EXPERT_sliced, cmap='RdBu', vmin=INPUT_EXPERT.min(), vmax=INPUT_EXPERT.max())
fig.colorbar(c, ax=ax)
plt.plot(y_m[:,0], y_m[:,2],'.')
plt.title('Expert - Control Value with Trajectories')
plt.show()

"""
Create animated mapping plots
"""

#Take only swingup part of mapping in external logic case
INPUT_list_swingup_ext_log = []

for j in range(0,len(history_ext_log.INPUT_list)):
    INPUT_list_swingup_ext_log.append(history_ext_log.INPUT_list[j][:, 0:ind_balSwing]) 
X_swingup = X[:, 0:ind_balSwing]
Y_swingup = Y[:, 0:ind_balSwing]

#Take only swingup part of mapping in internal logic case
INPUT_list_swingup_int_log = []
for j in range(0,len(history_int_log.INPUT_list)):
    INPUT_list_swingup_int_log.append(history_int_log.INPUT_list[j][:, 0:ind_balSwing]) 

#Animate mappings
bot.animate_mapping(history_ext_log.X, history_ext_log.Y, history_ext_log.INPUT_list, 'External Logic')
bot.animate_mapping(history_int_log.X, history_int_log.Y, history_int_log.INPUT_list, 'Internal Logic')
bot.animate_mapping(X_swingup, Y_swingup, INPUT_list_swingup_ext_log, 'SwingUp External Logic')
bot.animate_mapping(X_swingup, Y_swingup, INPUT_list_swingup_int_log, 'SwingUp Internal Logic')

#Take only swingup part of mapping in expert case
INPUT_EXPERT_swingup = INPUT_EXPERT[:, 0:ind_balSwing]
# Display the resulting image with pcolor()
INPUT_EXPERT_swingup_sliced = INPUT_EXPERT_swingup[:-1, :-1]

#Plot mapping only for swingup part in expert case
fig = plt.figure()
ax = plt.axes()
c = ax.pcolor(X_swingup, Y_swingup, INPUT_EXPERT_swingup_sliced, cmap='RdBu', vmin=INPUT_EXPERT_swingup_sliced.min(), vmax=INPUT_EXPERT_swingup_sliced.max())
fig.colorbar(c, ax=ax)
plt.title('Expert SwingUp - Control Value')
plt.show()


################################
#######Experimental ############
################################

"""
Experiment with DAGGER (manually)
1.) Look at animation
2.) Take trajectory and cut out part were controller fails
3.) Get input for cut trajectory from expert
"""
###############################################################################
#bot.animate_cart(y_lstm_temp)  
#y_lstm_temp = y_lstm_temp[400:600,:]
#bot.time = time[400:600]
#u_lstm_temp = bot.expert(y_lstm_temp)
#u_lstm = np.concatenate((u_lstm,u_lstm_temp), axis = 0)
#y_lstm_temp = nn_tools.sampleData(y_lstm_temp, look_back, Ts)
#y_lstm = np.concatenate((y_lstm,y_lstm_temp), axis = 0)
#time_pkg.sleep(2)
###############################################################################

#################################
#########Saving Data#############
#################################
    
#np.savetxt("trainLoop.csv", store, delimiter=',', fmt='%5.2f')


#import pickle
#d = { "abc" : [1, 2, 3], "qwerty" : [4,5,6] }
#afile = open(r'test.txt', 'wb')
#pickle.dump(d, afile)
#afile.close()
#
#file2 = open(r'test.txt', 'rb')
#new_d = pickle.load(file2)
#file2.close()