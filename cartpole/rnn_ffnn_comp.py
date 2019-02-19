# Force keras to use the CPU because it's actually faster for this size network
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' 
os.environ["CUDA_VISIBLE_DEVICES"] = ''

import numpy as np

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
from IPython import get_ipython
ipython = get_ipython()
ipython.magic('matplotlib qt')

from cartpole_class import Cartpole 
import nn_tools
import time as time_pkg
plt.close('all')
## Define some constants

# time vector, doesn't actually affect the integration, only what times it records our state variable at
dt = 0.01
time = np.arange(0.0, 20, dt)
u_max = 1000
# Cartpole is a class we defined that takes care of the simulation/animation of the cartpole
bot = Cartpole(time, u_max)
[y, u] = bot.simulate_ES_control()
#bot.animate_cart(y)
plt.figure()
plt.plot(time, y[:,1])
plt.plot(time, y[:,0])
plt.legend(['Pos', 'Angle'])
plt.title("u_max %2.3f " % u_max)
bot.verbose = 1
Ts = 1
look_back = 3
ys = nn_tools.sampleData(y[:,:,0], look_back, Ts)
store = np.zeros((10,8))
u_ff = u
u_fflb = u
u_lstm = u
y_ff = y
y_fflb = ys
y_lstm = ys


for i in range(0,10):
 
    bot.u_max = 1000
    
    #Train Feedforward network
    net_type = 'FF'
    ff_model = nn_tools.createNet(net_type)
    ff_history = ff_model.fit(y_ff.squeeze(), u_ff, batch_size = y_ff.shape[0], epochs=4000, verbose=1)
    #nn_tools.plotTrain(ff_history)
    [y_ff, u_ff_sim] = bot.simulate_NN_control(ff_model, net_type, Ts)
    print('FF',i, 'loss', ff_history.history['loss'][-1])

    #Train Feedforward network with looking back
    net_type = 'FFLB'
    fflb_model = nn_tools.createNet(net_type, look_back)
    fflb_history = fflb_model.fit(y_fflb, u_fflb, batch_size = y_fflb.shape[0], epochs=4000, verbose=1)
    #nn_tools.plotTrain(fflb_history)
    [y_fflb, u_fflb_sim] = bot.simulate_NN_control(fflb_model, net_type, Ts, look_back)
    print('FFLB',i, 'loss', fflb_history.history['loss'][-1])
    
    #Train LSTM
    net_type = 'LSTM'
    lstm_model = nn_tools.createNet(net_type, look_back)
    lstm_history = lstm_model.fit(np.swapaxes(y_lstm,1,2), u_lstm, batch_size = y_lstm.shape[0], epochs=600, verbose=1)
    #nn_tools.plotTrain(lstm_history)
    [y_lstm, u_lstm_sim] = bot.simulate_NN_control(lstm_model, net_type, Ts, look_back)
    print('LSTM',i, 'loss', lstm_history.history['loss'][-1])
    
    Y = np.zeros((len(time),4,4,1))
    Y[:,:,0,0] = y.squeeze()
    Y[:,:,1,0] = y_ff
    Y[:,:,2,0] = y_fflb
    Y[:,:,3,0] = y_lstm
    LABEL_ROWS = ['ES','FF','FFLB','LSTM']
    LABEL_COLS = ['CTRL_TYPE']
    info = [['' for x in range(1)] for y in range(4)]
    bot.animate_cart_dim(Y, LABEL_ROWS,LABEL_COLS, info)

    print('ES max(u)', np.max(np.abs(u)))
    print('FF max(u)', np.max(np.abs(u_ff_sim)))
    print('FFLB max(u)', np.max(np.abs(u_fflb_sim)))
    print('LSTM max(u)', np.max(np.abs(u_lstm_sim)))

    print('ES t_err', bot.calc_feat(y.squeeze()))
    print('FF t_err', bot.calc_feat(y_ff))
    print('FFLB t_err', bot.calc_feat(y_fflb))
    print('LSTM t_err', bot.calc_feat(y_lstm))

    store[i,0] = np.max(np.abs(u))
    store[i,1] = np.max(np.abs(u_ff))
    store[i,2] = np.max(np.abs(u_fflb))
    store[i,3] = np.max(np.abs(u_lstm))

    store[i,4] = bot.calc_feat(y.squeeze())
    store[i,5] = bot.calc_feat(y_ff)
    store[i,6] = bot.calc_feat(y_fflb)
    store[i,7] = bot.calc_feat(y_lstm)
    print(i)
   







################################
#######Experimental ############
################################





   # u_ff = bot.expert(y_ff)
   # u_fflb = bot.expert(y_fflb)
#    u_ff_temp = bot.expert(y_ff_temp)
#    u_ff = np.concatenate((u_ff,u_ff_temp), axis = 0)
#    y_ff_temp = y_ff_temp[:,:,np.newaxis]
#    y_ff = np.concatenate((y_ff,y_ff_temp), axis = 0)

#bot.animate_cart(y_lstm_temp)  
#y_lstm_temp = y_lstm_temp[400:600,:]
#bot.time = time[400:600]
#u_lstm_temp = bot.expert(y_lstm_temp)
#u_lstm = np.concatenate((u_lstm,u_lstm_temp), axis = 0)
#y_lstm_temp = nn_tools.sampleData(y_lstm_temp, look_back, Ts)
#y_lstm = np.concatenate((y_lstm,y_lstm_temp), axis = 0)
#time_pkg.sleep(2)

    
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