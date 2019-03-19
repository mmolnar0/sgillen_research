import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Flatten, Input, concatenate, Reshape, Lambda, Maximum, Multiply, Subtract, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
from math import pi

def createNet(netType, look_back = []):
     if(netType == 'FF'):
        # Feed forward 
        with tf.variable_scope('pi/simple_pol/'):
            ff_model = Sequential()
            ff_model.add(Dense(32, input_shape=(4,), activation ='relu'))
            ff_model.add(Dense(16, input_shape=(4,), activation ='relu'))           
            ff_model.add(Dense(1))
            adam = Adagrad(lr=0.1, epsilon=None, decay=0.0)
            ff_model.compile(loss='mean_squared_error', optimizer=adam)
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
         ##############MAYBE RELU#######################
        with tf.variable_scope('pi/simple_pol/'):
            lstm_model = Sequential()
            lstm_model.add(LSTM(12*look_back, input_shape =(look_back, 4)))
            lstm_model.add(Dense(12*look_back))
            lstm_model.add(Dense(1))
            lstm_model.compile(loss='mean_squared_error', optimizer='adam')
            return lstm_model
     elif(netType == 'Parallel'):
        with tf.variable_scope('pi/simple_pol/'):
            
            """
            Simple FF or Parallel with LSTM, with internal ref generation 
            """
#            # This returns a tensor
#            inputs = Input(shape=(look_back,4))
#            
#            n00 = Flatten()(inputs)
#            
#            n01 = Dense(12*look_back)(n00)
#            n02 = Dense(12*look_back)(n01)
#            n03 = Dense(12*look_back)(n02)
#            nRef = Dense(4*look_back)(n03)
#            
#            nSub = Subtract()([nRef,n00])            
##            nSubRe = Reshape((look_back,4))(nSub)
#            
#            n11 = Dense(12*look_back, name = 'n11')(nSub)
#            n12 = Dense(1, name = 'n12')(n11)
#            
#
##            n21 = LSTM(12*look_back, input_shape =(look_back, 4), name = 'n21')(nSubRe)
##            n22 = Dense(12*look_back, name = 'n22')(n21)
##            n23 = Dense(1, name = 'n23')(n22)
##            ncat = concatenate([n12,n23])    
##            
##            #Combine Information
##            ncat1 = Dense(12*look_back, name = 'n31')(ncat)
##            ncat2 = Dense(12*look_back, name = 'n32')(ncat1)         
##            ncat3 = Dense(12*look_back, name = 'n33')(ncat2)  
##            nout = Dense(1)(ncat3)
#
#            parallel_model = Model(inputs=inputs, outputs=n12)
#            parallel_model.compile(loss='mean_squared_error', optimizer='Adam')
#            return parallel_model
        
#            """
#            Simple FF with ref err as input
#            """
#            # This returns a tensor
#            inputs = Input(shape=(look_back,4))
#
#            def createRefBatch(inputs):
#                ref = np.array([[pi,0,0,0],
#                                [pi,0,0,0],
#                                [pi,0,0,0]])
#                refKeras = K.constant(ref.reshape((1,look_back,4)))
#                return tf.math.subtract(refKeras,inputs)
#            diff = Lambda(createRefBatch)(inputs)
#
#            n00 = Flatten()(diff)
#
#            n11 = Dense(12*look_back, name = 'n11')(n00)
#            n12 = Dense(1, name = 'n12')(n11)
#            
#            parallel_model = Model(inputs=inputs, outputs=n12)
#            parallel_model.compile(loss='mean_squared_error', optimizer='Adam')
#            return parallel_model      
        
            """
            Parallel with LSTM with parallel logic and ref err as input
            """
#            # This returns a tensor
#            inputs = Input(shape=(look_back,4))
#
#            def createRefBatch(inputs):
#                ref = np.array([[pi,0,0,0],
#                                [pi,0,0,0],
#                                [pi,0,0,0]])
#                refKeras = K.constant(ref.reshape((1,look_back,4)))
#                return tf.math.subtract(refKeras,inputs)
#            diff = Lambda(createRefBatch)(inputs)
#
#            n00 = Flatten()(diff)
#            
#            
#            nlog1 =  Dense(12*look_back, name = 'nlog1', activation ='sigmoid')(n00)
#            nlog2 =  Dense(12*look_back, name = 'nlog2', activation ='sigmoid')(nlog1)
#            nlog3 =  Dense(2, name = 'nlog3', activation ='softmax')(nlog2)
#
#            n11 = Dense(12*look_back, name = 'n11')(n00)
#            n12 = Dense(1, name = 'n12')(n11)
#
#            n21 = LSTM(12*look_back, input_shape =(look_back, 4), name = 'n21')(inputs)
#            n22 = Dense(12*look_back, name = 'n22')(n21)
#            n23 = Dense(1, name = 'n23')(n22)
#            ncat = concatenate([n12,n23])    
#            
#            #Combine Information
#            noutLog = Multiply()([ncat,nlog3])
#            nout = Dense(1)(noutLog)
#
#            parallel_model = Model(inputs=inputs, outputs=nout)
#            parallel_model_inspect = Model(inputs=inputs, outputs=nlog3)
#            parallel_model.compile(loss='mean_squared_error', optimizer='Adam')
#            parallel_model_inspect.compile(loss='mean_squared_error', optimizer='Adam')            
#            return parallel_model, parallel_model_inspect      
        
            """
            Parallel with LSTM and ref err as input 
            """            
#            # This returns a tensor
#            inputs = Input(shape=(look_back,4))
#
#            def createRefBatch(inputs):
#                ref = np.array([[pi,0,0,0],
#                                [pi,0,0,0],
#                                [pi,0,0,0]])
#                refKeras = K.constant(ref.reshape((1,look_back,4)))
#                return tf.math.subtract(refKeras,inputs)
#            diff = Lambda(createRefBatch)(inputs)
#
#            n00 = Flatten()(diff)
#            
#
#            n11 = Dense(12*look_back, name = 'n11')(n00)
#            n12 = Dense(1, name = 'n12')(n11)
#            
#
#            n21 = LSTM(12*look_back, input_shape =(look_back, 4), name = 'n21')(inputs)
#            n22 = Dense(12*look_back, name = 'n22')(n21)
#            n23 = Dense(1, name = 'n23')(n22)
#            
#
#            ncat = concatenate([n12,n23])    
#            
#            #Combine Information
#            ncat1 = Dense(12*look_back, name = 'n31')(ncat)
#            ncat2 = Dense(12*look_back, name = 'n32')(ncat1)            
#            nout = Dense(1)(ncat2)
#
#            parallel_model = Model(inputs=inputs, outputs=nout)
#            parallel_model.compile(loss='mean_squared_error', optimizer='Adam')
#            return parallel_model
            
        
            """
            Parallel with parallel logic
            """
#            # This returns a tensor
#            inputs = Input(shape=(look_back,4))
#            n00 = Flatten()(inputs)
#            
#            nlog1 =  Dense(12*look_back, name = 'nlog1', activation ='sigmoid')(n00)
#            nlog2 =  Dense(12*look_back, name = 'nlog2', activation ='sigmoid')(nlog1)
#            nlog3 =  Dense(2, name = 'nlog3', activation ='softmax')(nlog2)
#
#            n11 = Dense(12*look_back, name = 'n11')(n00)
#            n12 = Dense(1, name = 'n12')(n11)
#            
#
#            n21 = Dense(12*look_back, name = 'n21')(n00)
#            n22 = Dense(12*look_back, name = 'n22')(n21)
#            n23 = Dense(1, name = 'n23')(n22)
#            ncat = concatenate([n12,n23])    
#            
#            #Combine Information
#            noutLog = Multiply()([ncat,nlog3])
#            nout = Dense(1)(noutLog)
#
#            parallel_model = Model(inputs=inputs, outputs=nout)
#            parallel_model.compile(loss='mean_squared_error', optimizer='Adam')
#            return parallel_model
            
            """
            Parallel with FF with parallel logic and ref err as input
#            """
#            # This returns a tensor
#            inputs = Input(shape=(4,))
#
#            def createRefBatch(inputs):
#                ref = np.array([pi,0,0,0])
#                refKeras = K.constant(ref.reshape((1,4)))
#                return tf.math.subtract(refKeras,inputs)
#            diff = Lambda(createRefBatch)(inputs)            
#            
#            nlog1 =  Dense(12, name = 'nlog1', activation ='sigmoid')(diff)
#            nlog2 =  Dense(12, name = 'nlog2', activation ='sigmoid')(nlog1)
#            nlog3 =  Dense(2, name = 'nlog3', activation ='softmax')(nlog2)
#
#            n11 = Dense(12, name = 'n11')(diff)
#            n12 = Dense(1, name = 'n12')(n11)
#            
#            n21 = Dense(12, name = 'n21', activation ='relu')(diff)
#            n22 = Dense(12, name = 'n22', activation ='relu')(n21)
#            n23 = Dense(1, name = 'n23')(n22)
#            ncat = concatenate([n12,n23])    
#            
#            #Combine Information
#            noutLog = Multiply()([ncat,nlog3])
#            nout = Dense(1)(noutLog)
#
#            parallel_model = Model(inputs=inputs, outputs=nout)
#            parallel_model_inspect = Model(inputs=inputs, outputs=nlog3)
#            parallel_model.compile(loss='mean_squared_error', optimizer='Adam')
#            parallel_model_inspect.compile(loss='mean_squared_error', optimizer='Adam')            
#            return parallel_model, parallel_model_inspect      
        
            """
            Parallel with FF with parallel logic (as input) and ref err as input
            """
#            # This returns a tensor
#            inputs = Input(shape=(4,))
#            inputsLog = Input(shape=(2,))
#            def createRefBatch(inputs):
#                ref = np.array([pi,0,0,0])
#                refKeras = K.constant(ref.reshape((1,4)))
#                return tf.math.subtract(refKeras,inputs)
#            diff = Lambda(createRefBatch)(inputs)            
#
#            n11 = Dense(12, name = 'n11')(diff)
#            n12 = Dense(1, name = 'n12')(n11)
#            
#            n21 = Dense(12, name = 'n21', activation ='relu')(diff)
#            n22 = Dense(12, name = 'n22', activation ='relu')(n21)
#            n23 = Dense(1, name = 'n23')(n22)
#            ncat = concatenate([n12,n23])    
#            
#            #Combine Information
#            noutLog = Multiply()([ncat,inputsLog])
#            nout = Reshape(target_shape=(1,))(Lambda(lambda x : K.sum(x, axis=1)) (noutLog))
#
#            parallel_model = Model(inputs=[inputs, inputsLog], outputs=nout)
#            parallel_model.compile(loss='mean_squared_error', optimizer='Adam')         
#            return parallel_model    

            """
            Parallel with LSTM, parallel logic (as input and internally), and ref err as input
            """
            # This returns a tensor
            inputs = Input(shape=(look_back,4))
            inputsLog = Input(shape=(2,))

            def createRefBatch(inputs):
                ref = np.array([[pi,0,0,0],
                                [pi,0,0,0],
                                [pi,0,0,0]])
                refKeras = K.constant(ref.reshape((1,look_back,4)))
                return tf.math.subtract(refKeras,inputs)
            diff = Lambda(createRefBatch)(inputs)       

            n00 = Flatten()(diff)
            
            n11 = Dense(12*look_back, name = 'n11')(n00)
            n12 = Dense(1, name = 'n12')(n11)
            
            n21 = LSTM(12*look_back, input_shape =(look_back, 4), name = 'n21')(inputs)
            n22 = Dense(12*look_back, name = 'n22')(n21)
            n23 = Dense(1, name = 'n23')(n22)
            
            
            ncat = concatenate([n12,n23])                         

            #Combine Information
            noutLogExt = Multiply()([ncat,inputsLog])
            nout1 = Reshape(target_shape=(1,))(Lambda(lambda x : K.sum(x, axis=1)) (noutLogExt))

            nlog1 =  Dense(12, name = 'nlog1', activation ='sigmoid')(n00)
            nlog2 =  Dense(12, name = 'nlog2', activation ='sigmoid')(nlog1)
            nlog3 =  Dense(2, name = 'nlog3', activation ='softmax')(nlog2)
            
            #Combine Information
            noutLogInt = Multiply()([ncat,nlog3])
            nout2 = Reshape(target_shape=(1,))(Lambda(lambda x : K.sum(x, axis=1))(noutLogInt))

            parallel_model1 = Model(inputs=[inputs, inputsLog], outputs=nout1)
            parallel_model2 = Model(inputs= inputs, outputs=nout2)
            parallel_model_inspect = Model(inputs=inputs, outputs=nlog3)
            parallel_model1.compile(loss='mean_squared_error', optimizer='Adam')         
            parallel_model2.compile(loss='mean_squared_error', optimizer='Adam')      
            return parallel_model1, parallel_model2, parallel_model_inspect   
        
        
        
#                    # This returns a tensor
#            inputs = Input(shape=(4,))
#            inputsLog = Input(shape=(2,))
#            def createRefBatch(inputs):
#                ref = np.array([pi,0,0,0])
#                refKeras = K.constant(ref.reshape((1,4)))
#                return tf.math.subtract(refKeras,inputs)
#            diff = Lambda(createRefBatch)(inputs)            
#
#            n11 = Dense(12, name = 'n11')(diff)
#            n12 = Dense(1, name = 'n12')(n11)
#            
#            n21 = Dense(12, name = 'n21', activation ='relu')(diff)
#            n22 = Dense(12, name = 'n22', activation ='relu')(n21)
#            n23 = Dense(1, name = 'n23')(n22)
#            ncat = concatenate([n12,n23])    
#            
#            #Combine Information
#            noutLogExt = Multiply()([ncat,inputsLog])
#            nout1 = Reshape(target_shape=(1,))(Lambda(lambda x : K.sum(x, axis=1)) (noutLogExt))
#
#            nlog1 =  Dense(12, name = 'nlog1', activation ='sigmoid')(diff)
#            nlog2 =  Dense(12, name = 'nlog2', activation ='sigmoid')(nlog1)
#            nlog3 =  Dense(2, name = 'nlog3', activation ='softmax')(nlog2)
#            
#            #Combine Information
#            noutLogInt = Multiply()([ncat,nlog3])
#            nout2 = Reshape(target_shape=(1,))(Lambda(lambda x : K.sum(x, axis=1))(noutLogInt))
#
#            parallel_model1 = Model(inputs=[inputs, inputsLog], outputs=nout1)
#            parallel_model2 = Model(inputs= inputs, outputs=nout2)
#            parallel_model_inspect = Model(inputs=inputs, outputs=nlog3)
#            parallel_model1.compile(loss='mean_squared_error', optimizer='Adam')         
#            parallel_model2.compile(loss='mean_squared_error', optimizer='Adam')      
#            return parallel_model1, parallel_model2, parallel_model_inspect   

        
     elif(netType == 'FFSimple'):
            # Feed forward 
            with tf.variable_scope('pi/simple_pol/'):
                ff_model = Sequential()
                ff_model.add(Dense(1))
                adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                ff_model.compile(loss='mean_squared_error', optimizer=adam)
                return ff_model
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

def getControlMesh_extLog(bot, model, theta_lim = [-pi*1.2, pi*1.2], theta_dot_lim = [-1, 1], resolution = 100):
    """ Generates map for control input over theta and theta_dot with external logic network and expert controller
    """
    # Generate two 1-D arrays: u, v
    u = np.linspace(theta_lim[0], theta_lim[1], resolution)
    v = np.linspace(theta_dot_lim[0], theta_dot_lim[1], resolution)
    
    # Generate 2-D arrays from u and v: X, Y
    X,Y = np.meshgrid(u, v)

    INPUT = np.zeros((u.size, v.size))
    INPUT_EXPERT = np.zeros((u.size, v.size))

    for i in range(v.size):
        for j in range(u.size):
            if ((u[j] < (140 * (pi/180)) ) or (u[j] > (220 * (pi/180)))):
                switch_log = np.array([[0,1]])
            else:
                switch_log = np.array([[1,0]])      
            out = model.predict([np.array([[u[j], 0, v[i], 0]]), switch_log])
            INPUT[i,j] = out[0]    
            INPUT_EXPERT[i,j] = bot.controlES(np.array([u[j], 0, v[i], 0]))
    return X, Y, INPUT, INPUT_EXPERT

def getControlMesh_IntLog(bot, model, model_inspect, theta_lim = [-pi*1.2, pi*1.2], theta_dot_lim = [-7, 7], resolution = 100):
    """ Generates map for control input over theta and theta_dot with internal logic network and expert controller
    """
    # Generate two 1-D arrays: u, v
    u = np.linspace(theta_lim[0], theta_lim[1], resolution)
    v = np.linspace(theta_dot_lim[0], theta_dot_lim[1], resolution)
    
    # Generate 2-D arrays from u and v: X, Y
    X,Y = np.meshgrid(u, v)
    L1 = np.zeros((u.size, v.size))
    L2 = np.zeros((u.size, v.size))
    INPUT = np.zeros((u.size, v.size))
    INPUT_EXPERT = np.zeros((u.size, v.size))
    # Compute Z based on X and Y
    for i in range(v.size):
        for j in range(u.size):        
            out = model_inspect.predict(np.array([[u[j], 0, v[i], 0]]))
            L1[i,j] = out[0,0]
            L2[i,j] = out[0,1]
            out = model.predict(np.array([[u[j], 0, v[i], 0]]))
            INPUT[i,j] = out[0]
            
            INPUT_EXPERT[i,j] = bot.controlES(np.array([u[j], 0, v[i], 0]))
    return X, Y, INPUT, INPUT_EXPERT, L1, L2
"""
Same two functions for mappings as above only adapted for using a neural network model with look back
"""
def getControlMesh_lb_extLog(bot, model, theta_lim = [-pi*1.2, pi*1.2], theta_dot_lim = [-1, 1], resolution = 100):
    """ Generates map for control input over theta and theta_dot with internal logic network and expert controller
    """
    # Generate two 1-D arrays: u, v
    u = np.linspace(theta_lim[0], theta_lim[1], resolution)
    v = np.linspace(theta_dot_lim[0], theta_dot_lim[1], resolution)
    
    # Generate 2-D arrays from u and v: X, Y
    X,Y = np.meshgrid(u, v)

    INPUT = np.zeros((u.size, v.size))
    INPUT_EXPERT = np.zeros((u.size, v.size))
  
    for i in range(v.size):
        for j in range(u.size):
            if ((u[j] < (140 * (pi/180)) ) or (u[j] > (220 * (pi/180)))):
                switch_log = np.array([[0,1]])
            else:
                switch_log = np.array([[1,0]])      
            out = model.predict([np.array([[[u[j], 0, v[i], 0],[u[j], 0, v[i], 0],[u[j], 0, v[i], 0]]]), switch_log])
            INPUT[i,j] = out[0]    
            INPUT_EXPERT[i,j] = bot.controlES(np.array([u[j], 0, v[i], 0]))
    return X, Y, INPUT, INPUT_EXPERT

def getControlMesh_lb_IntLog(bot, model, model_inspect, theta_lim = [-pi*1.2, pi*1.2], theta_dot_lim = [-7, 7], resolution = 100):
    """ Generates map for control input over theta and theta_dot with internal logic network and expert controller
    """
    # Generate two 1-D arrays: u, v
    u = np.linspace(theta_lim[0], theta_lim[1], resolution)
    v = np.linspace(theta_dot_lim[0], theta_dot_lim[1], resolution)
    
    # Generate 2-D arrays from u and v: X, Y
    X,Y = np.meshgrid(u, v)
    L1 = np.zeros((u.size, v.size))
    L2 = np.zeros((u.size, v.size))
    INPUT = np.zeros((u.size, v.size))
    INPUT_EXPERT = np.zeros((u.size, v.size))

    for i in range(v.size):
        for j in range(u.size):        
            out = model_inspect.predict(np.array([[[u[j], 0, v[i], 0],[u[j], 0, v[i], 0],[u[j], 0, v[i], 0]]]))
            L1[i,j] = out[0,0]
            L2[i,j] = out[0,1]
            out = model.predict(np.array([[[u[j], 0, v[i], 0],[u[j], 0, v[i], 0],[u[j], 0, v[i], 0]]]))
            INPUT[i,j] = out[0]
            
            INPUT_EXPERT[i,j] = bot.controlES(np.array([u[j], 0, v[i], 0]))
    return X, Y, INPUT, INPUT_EXPERT, L1, L2

class InputHistory_extLog(keras.callbacks.Callback):
    """ Callback class for external logic network training
    """
    def on_train_begin(self, logs={}):
        self.INPUT_list = []
        
        # Generate two 1-D arrays: u, v
        self.u = np.linspace(-pi*1.2, pi*1.2, 100)
        self.v = np.linspace(-1, 1, 100)
        self.INPUT = np.zeros((self.u.size, self.v.size))
        # Generate 2-D arrays from u and v: X, Y
        self.X, self.Y = np.meshgrid(self.u, self.v)
        self.prev_loss = 1e9
        
    #After every epoch save current mapping in INPUT_list      
    def on_epoch_end(self, epoch, logs={}):  
        if ((self.prev_loss-logs.get('loss'))/self.prev_loss) > 0.5:
            self.prev_loss = logs.get('loss')
            self.INPUT = np.zeros((self.u.size, self.v.size))
            # Compute Z based on X and Y
            for i in range(self.v.size):
                for j in range(self.u.size):
                    if ((self.u[j] < (140 * (pi/180)) ) or (self.u[j] > (220 * (pi/180)))):
                        switch_log_temp = np.array([[0,1]])
                    else:
                        switch_log_temp = np.array([[1,0]])      
                    out = self.model.predict([np.array([[self.u[j], 0, self.v[i], 0]]), switch_log_temp])
                    self.INPUT[i,j] = out[0]
            self.INPUT_list.append(self.INPUT)
            
class InputHistory(keras.callbacks.Callback):
    """ Callback class for internal logic network training
    """
    def on_train_begin(self, logs={}):
        self.INPUT_list = []        #For input maps which are appended to the list everytime when loss gets halved
        
        # Generate two 1-D arrays: u, v
        self.u = np.linspace(-pi*1.2, pi*1.2, 100)
        self.v = np.linspace(-1, 1, 100)
        self.INPUT = np.zeros((self.u.size, self.v.size))
        # Generate 2-D arrays from u and v: X, Y
        self.X, self.Y = np.meshgrid(self.u, self.v)
        self.prev_loss = 1e9      
        
    #After every epoch save current mapping in INPUT_list  
    def on_epoch_end(self, epoch, logs={}):  
        if ((self.prev_loss-logs.get('loss'))/self.prev_loss) > 0.5:
            self.prev_loss = logs.get('loss')
            self.INPUT = np.zeros((self.u.size, self.v.size))
            # Compute Z based on X and Y
            for i in range(self.v.size):
                for j in range(self.u.size): 
                    out = self.model.predict(np.array([[self.u[j], 0, self.v[i], 0]]))
                    self.INPUT[i,j] = out[0]
            self.INPUT_list.append(self.INPUT)
     
"""
Same two functions for saving mappings during training as above only adapted for using a neural network model with look back
""" 
class InputHistory_lb_extLog(keras.callbacks.Callback):
    """ Callback class for external logic network training
    """
    def on_train_begin(self, logs={}):
        self.INPUT_list = []
        
        # Generate two 1-D arrays: u, v
        self.u = np.linspace(-pi*1.2, pi*1.2, 100)
        self.v = np.linspace(-1, 1, 100)
        self.INPUT = np.zeros((self.u.size, self.v.size))
        # Generate 2-D arrays from u and v: X, Y
        self.X, self.Y = np.meshgrid(self.u, self.v)
        self.prev_loss = 1e9
        
    def on_epoch_end(self, epoch, logs={}):  
        if ((self.prev_loss-logs.get('loss'))/self.prev_loss) > 0.5:
            self.prev_loss = logs.get('loss')
            self.INPUT = np.zeros((self.u.size, self.v.size))
            # Compute Z based on X and Y
            for i in range(self.v.size):
                for j in range(self.u.size):
                    if ((self.u[j] < (140 * (pi/180)) ) or (self.u[j] > (220 * (pi/180)))):
                        switch_log_temp = np.array([[0,1]])
                    else:
                        switch_log_temp = np.array([[1,0]])      
                    out = self.model.predict([np.array([[[self.u[j], 0, self.v[i], 0],[self.u[j], 0, self.v[i], 0],[self.u[j], 0, self.v[i], 0]]]), switch_log_temp])
                    self.INPUT[i,j] = out[0]
            self.INPUT_list.append(self.INPUT)
            
class InputHistory_lb(keras.callbacks.Callback):
    """ Callback class for internal logic network training
    """
    def on_train_begin(self, logs={}):
        self.INPUT_list = []
        self.prev_loss = 1e9
        # Generate two 1-D arrays: u, v
        self.u = np.linspace(-pi*1.2, pi*1.2, 100)
        self.v = np.linspace(-1, 1, 100)
        self.INPUT = np.zeros((self.u.size, self.v.size))
        # Generate 2-D arrays from u and v: X, Y
        self.X, self.Y = np.meshgrid(self.u, self.v)
        
    def on_epoch_end(self, epoch, logs={}):  
        if ((self.prev_loss-logs.get('loss'))/self.prev_loss) > 0.5:
            self.prev_loss = logs.get('loss')
            self.INPUT = np.zeros((self.u.size, self.v.size))
            # Compute Z based on X and Y
            for i in range(self.v.size):
                for j in range(self.u.size): 
                    out = self.model.predict(np.array([[[self.u[j], 0, self.v[i], 0],[self.u[j], 0, self.v[i], 0],[self.u[j], 0, self.v[i], 0]]]))
                    self.INPUT[i,j] = out[0]
            self.INPUT_list.append(self.INPUT)

        
        
        
        