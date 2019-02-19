import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
import matplotlib.pyplot as plt
import numpy as np

def createNet(netType, look_back = []):
     if(netType == 'FF'):
        # Feed forward 
        with tf.variable_scope('pi/simple_pol/'):
            ff_model = Sequential()
            ff_model.add(Dense(12, input_shape=(4,), activation ='relu'))
            ff_model.add(Dense(12, input_shape=(4,), activation ='relu'))
            ff_model.add(Dense(1))
            ff_model.compile(loss='mean_squared_error', optimizer='adam')
            return ff_model
     elif(netType == 'FFLB'):
         # Feed forward with looking back
        with tf.variable_scope('pi/simple_pol/'):
            fflb_model = Sequential()
            fflb_model.add(Flatten())
            fflb_model.add(Dense(12*look_back, activation ='relu'))
            fflb_model.add(Dense(12*look_back, activation ='relu'))
            fflb_model.add(Dense(1))
            fflb_model.compile(loss='mean_squared_error', optimizer='adam')
            return fflb_model
     elif(netType == 'LSTM'):
         # Long short-term memory
        with tf.variable_scope('pi/simple_pol/'):
            lstm_model = Sequential()
            lstm_model.add(LSTM(12*look_back, input_shape =(look_back, 4)))
            lstm_model.add(Dense(12*look_back))
            lstm_model.add(Dense(1))
            lstm_model.compile(loss='mean_squared_error', optimizer='adam')
            return lstm_model

def plotTrain(nn_history):
    #Plot the loss over time
    plt.figure()
    plt.plot(nn_history.history['loss'])
    plt.title('simple model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
#Sample the energy shaping controller states
def sampleData(x, n, Ts):
    """splits up data array x into sets of length n with sampling time Ts
    x:  input matrix (M,N)
    n:  number of samples per set
    Ts: sample time
    """
    N = x.shape[0]                                              #number of input vectors
    z_ext = np.zeros(((n-1)*Ts,x.shape[1]))
    x = np.concatenate((z_ext, x), axis=0)
    #calculate number of sets
    nset = N
    y = np.zeros((nset,)+(x.shape[1],)+(n,))                    #initialize output matrix
    step = 0      
    #iterate through input data                                              
    while(step<nset):
        #select vectors according to n and Ts
        y[step,:,:] = np.transpose(x[step:(n-1)*Ts+1+step:Ts,:]) 
        step+=1                                                 
    return y


        
        
        
        