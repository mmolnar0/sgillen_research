import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate

from numba import jit
import time


class Cartpole:
    """
    Implements dynamics, animation, and control for for a simple cartpole pendulum.

    Meant as a testbed for different controllers, the default controller (implemented in control) does a pretty good job
    though.

    The default constructor just goes, I find it more convenient to just go ahead and construct an object and then c
    change parameters after the fact.

    I.E.

    cart = Cartpole()
    cart.L = 5.0

    Attributes:
        L - length of the pendulum in (m)
        mc - mass of the kart (kg)
        mp - magnitude of pointmass at the end of the cart's pole (kg)
        g - force f gravity (N)
    
    """

    # Define constants (geometry and mass properties):
    def __init__(self, time, u_max):
        self.L = 1.0;  # length of the pole (m)
        self.mc = 4.0  # mass of the cart (kg)
        self.mp = 1.0  # mass of the ball at the end of the pole
        self.time = time
        self.dt = time[1]-time[0]
        self.g = 9.8;
        self.u_max = u_max
       
        self.Ts = []
        self.look_back = [];
        self.tNext = 0
        self.u_hold = []
        self.y_lb = []
        self.verbose = 0
        
        # State deviation considered as okay
        theta = pi*1.02
        x = 0.02
        th_dot = 0.02
        xdot = 0.02
    
        # Respective error (euclidean norm)
        final_state_dev = np.array([theta, x, th_dot, xdot])
        self.final_state = np.array([pi, 0, 0, 0])
        self.err_final = np.sqrt(np.sum((final_state_dev-self.final_state)**2))
    # TODO, refacter to switch t,y -> y,t to be consitent with the derivs
    def animate_cart(self, y):
        """
        constructs an animation object and returns it to the user.

        Then depending on your environment you'll need to do some other call to actually display the animation.
        usually I'm calling this from a jupyter notebook, in which case I do:



        ani = bot.animate_cart(time, y)
        HTML(ani.to_jshtml())



        :param t: numpy array with the time steps corresponding to the trajectory you want to animate, does not have to
        be uniform

        :param y: numpy array with a trajectory of state variables you want animated. [theta , x, thetadot, xdot]

        :return: matplotlib.animation, which you then need to display
        """

        x1 = y[:, 1]
        y1 = 0.0

        x2 = self.L * sin(y[:, 0]) + x1
        y2 = -self.L * cos(y[:, 0]) + y1

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, aspect='equal',
                             xlim=(-3, 3), ylim=(-3, 3))
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text


        def animate(i):
            thisx = [x1[i], x2[i]]
            thisy = [y1, y2[i]]

            line.set_data(thisx, thisy)
            time_text.set_text(time_template % (i * self.dt))
            return line, time_text

        return animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=self.dt*1000, blit=True, init_func=init)

    # @jit(nopython=False)
    def control(self, q):
        """
        This is where you should define the control for the cartpole, called by derivs.

        By default, implements a swingup controller for the cartpole based on energy shaping. Switches to an LQR to
        balance the pendulum

        :param q: numpy array of state variables [theta, x, thetadot, xdot]
        :return: u, the control torque in N*m
        """

        if (q[0] < 140 * pi/180) or (q[0] > 220 * pi/180 ):
            # swing up
            # energy error: Ee
            Ee = 0.5 * self.mp * self.L * self.L * q[2] ** 2 - self.mp * self.g * self.L * (1 + cos(q[0]))
            # energy control gain:
            k = 0.23
            # input acceleration: A (of cart)
            A = k * Ee * cos(q[0]) * q[2]
            # convert A to u (using EOM)
            delta = self.mp * sin(q[0]) ** 2 + self.mc
            u = A * delta - self.mp * self.L * (q[2] ** 2) * sin(q[0]) - self.mp * self.g * sin(q[2]) * cos(q[2])
        else:
            # balancing
            # LQR: K values from MATLAB
            k1 = 140.560
            k2 = -3.162
            k3 = 41.772
            k4 = -8.314
            u = -(k1 * (q[0] - pi) + k2 * q[1] + k3 * q[2] + k4 * q[3])
        return min(self.u_max, max(-self.u_max,u))

    # state vector: q = transpose([theta, x, d(theta)/dt, dx/dt])
    # @jit(nopython=False)
    
    def derivs(self, q, t):
        """
        Implements the dynamics for our cartpole, you need to integrate this yourself with E.G:

        y = integrate.odeint(bot.derivs, init_state, time)

        or whatever other ode solver you prefer.


        :param q: numpy array of state variables [theta, x, thetadot, xdot]
        :param t: float with the current time (not actually used but most ODE solvers want to pass this in anyway)
        :return: numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]
        """
        if(self.verbose):
            print('Time', t)    
        dqdt = np.zeros_like(q)

        # control input
        u = self.control(q)
        
        self.u_hist.append(u)

        delta = self.mp * sin(q[0]) ** 2 + self.mc

        dqdt[0] = q[2]
        dqdt[1] = q[3]

        dqdt[2] = - self.mp * (q[2] ** 2) * sin(q[0]) * cos(q[0]) / delta \
                  - (self.mp + self.mc) * self.g * sin(q[0]) / delta / self.L \
                  - u * cos(q[0]) / delta / self.L

        dqdt[3] = self.mp * self.L * (q[2] ** 2) * sin(q[0]) / delta \
                  + self.mp * self.L * self.g * sin(q[0]) * cos(q[0]) / delta / self. L \
                  + u / delta
                  
        return dqdt
    
    def derivs_dig(self, q, t):
        """
        Implements the dynamics for our cartpole, you need to integrate this yourself with E.G:
    
        y = integrate.odeint(bot.derivs, init_state, time)
    
        or whatever other ode solver you prefer.
    
        :param q: numpy array of state variables [theta, x, thetadot, xdot]
        :param t: float with the current time (not actually used but most ODE solvers want to pass this in anyway)
        :return: numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]
        """
        if(self.verbose):
            print('Time', t)
        if(t>=self.tNext):    #<>
            self.tNext += self.Ts*self.dt
            self.u_hold = self.control(q)
            
        self.u_hist.append(self.u_hold)    
            
        dqdt = np.zeros_like(q)
        
        delta = self.mp * sin(q[0]) ** 2 + self.mc
    
        dqdt[0] = q[2]
        dqdt[1] = q[3]
        
        dqdt[2] = - self.mp * (q[2] ** 2) * sin(q[0]) * cos(q[0]) / delta \
                      - (self.mp + self.mc) * self.g * sin(q[0]) / delta / self.L \
                      - self.u_hold * cos(q[0]) / delta / self.L
    
        dqdt[3] = self.mp * self.L * (q[2] ** 2) * sin(q[0]) / delta \
                  + self.mp * self.L * self.g * sin(q[0]) * cos(q[0]) / delta / self. L \
                  + self.u_hold / delta
  
        return dqdt
    
    def derivs_dig_lb(self, q, t):
        """
        Implements the dynamics for our cartpole, you need to integrate this yourself with E.G:

        y = integrate.odeint(bot.derivs, init_state, time)

        or whatever other ode solver you prefer.


        :param q: numpy array of state variables [theta, x, thetadot, xdot]
        :param t: float with the current time (not actually used but most ODE solvers want to pass this in anyway)
        :return: numpy array with the derivatives of the current state variable [thetadot, xdot, theta2dot, x2dot]
        """
        if(self.verbose):
            print('Time', t)
        if(t==0):
            z_ext = np.zeros((q.size,(self.look_back-1)*self.Ts))
            self.y_lb = np.concatenate((z_ext, q[:,np.newaxis]), axis=1) 
            self.tNext = self.Ts*self.dt
            self.u_lb = self.control(self.y_lb)
        else:
            if(t>=self.tNext):    #<>
                self.tNext += self.Ts*self.dt
                self.y_lb = np.concatenate((self.y_lb[:,1:],q[:,np.newaxis]), axis=1)
                self.u_lb = self.control(self.y_lb)
        
        self.u_hist.append(self.u_lb)
                
        dqdt = np.zeros_like(q)
    
        delta = self.mp * sin(q[0]) ** 2 + self.mc

        dqdt[0] = q[2]
        dqdt[1] = q[3]
        
        dqdt[2] = - self.mp * (q[2] ** 2) * sin(q[0]) * cos(q[0]) / delta \
                      - (self.mp + self.mc) * self.g * sin(q[0]) / delta / self.L \
                      - self.u_lb * cos(q[0]) / delta / self.L
    
        dqdt[3] = self.mp * self.L * (q[2] ** 2) * sin(q[0]) / delta \
                  + self.mp * self.L * self.g * sin(q[0]) * cos(q[0]) / delta / self. L \
                  + self.u_lb / delta
  
        return dqdt

    def animate_cart_dim(self, Y, LABEL_ROWS,LABEL_COLS, info):
        """
        constructs an animation object and returns it to the user.

        Then depending on your environment you'll need to do some other call to actually display the animation.
        usually I'm calling this from a jupyter notebook, in which case I do:

        

        ani = bot.animate_cart(time, t, Y, LABEL_ROWS,LABEL_COLS, info)
        HTML(ani.to_jshtml())

        :param t: numpy array with the time steps corresponding to the trajectory you want to animate, does not have to
        be uniform

        :param Y: 4D-Numpy matrix with numpy arrays containing trajectories of state variables you want animated.
        Last two dimensions for selecting the numpy array trajectories [theta , x, thetadot, xdot] of dimension [Nx4] with N
        as the number of trajectory samples.
        
        :param LABEL_ROWS, LABEL_COLS: Row and Column labels for subplots grid
        
        :param info: 2D list with strings for having an info text at the respective position in the subplots

        :return: matplotlib.animation, which you then need to display
        """
        #Convert row and col to linear index
        def sub2ind(array_shape, rows, cols):
            return rows*array_shape[1] + cols +1
        
        dt = (self.time[-1] - self.time[0])/len(self.time)              #Get time step
        dim_sub = Y.shape[2:4]                  #shape of subplot grid
        N = Y.shape[0]                          #Number of samples
        Nplot = dim_sub[0]*dim_sub[1]           #Number of subplots
        
        #Initialize variables plot points, axes, lines and text
        X1 = np.zeros((N, dim_sub[0], dim_sub[1]))
        Y1 = 0
        X2 = np.zeros((N, dim_sub[0], dim_sub[1]))
        Y2 = np.zeros((N, dim_sub[0], dim_sub[1]))
        AX = [[0 for x in range(dim_sub[1])] for y in range(dim_sub[0])] 
        LINE = [0 for x in range(Nplot)]
        TIME_TEXT = [0 for x in range(Nplot)]
        time_template = 'time = %.1fs'    
        
        fig = plt.figure()
        
        #Iterate trough all trajectories and create plot points, subplots, info texts and time
        for i in range(0,dim_sub[0]):
            for j in range(0,dim_sub[1]):
                #Trajectories to plot points
                X1[:,i,j] = Y[:,1,i,j]
                X2[:,i,j] = self.L * sin(Y[:,0,i,j]) + X1[:,i,j]
                Y2[:,i,j] = -self.L * cos(Y[:,0,i,j]) + Y1
                #Subplots
                AX[i][j]  = fig.add_subplot(dim_sub[0], dim_sub[1], sub2ind(dim_sub,i,j), autoscale_on=False, aspect='equal',
                             xlim=(-3, 3), ylim=(-3, 3))
                AX[i][j].grid()
                #Labels for columns and rows
                if i==0:
                    AX[i][j].set_title(LABEL_COLS[j])
                if j==0:
                    AX[i][j].text(-0.2, 0.55, LABEL_ROWS[i], transform=AX[i][j].transAxes, rotation=90)
                #Create line objects
                LINE[sub2ind(dim_sub,i,j)-1], = AX[i][j].plot([], [], 'o-', lw=2)
                #Add info text to plot
                AX[i][j].text(0.05, 0.1, info[i][j], transform=AX[i][j].transAxes)
                #Add time text to plot
                TIME_TEXT[sub2ind(dim_sub,i,j)-1] = AX[i][j].text(0.05, 0.9, '', transform=AX[i][j].transAxes)
        #Append text objects to line objects (necessary for using matplotlib FuncAnimation)         
        LINE += TIME_TEXT
        #Init for FuncAnimation
        def init():
            
            for i in range(0,dim_sub[0]):
                for j in range(0,dim_sub[1]):
                    LINE[sub2ind(dim_sub,i,j)-1].set_data([],[])
                    LINE[Nplot+sub2ind(dim_sub,i,j)-1].set_text('')
            return LINE
        #During animation adapt plot and time text
        def animate(k):

            for i in range(0,dim_sub[0]):
                for j in range(0,dim_sub[1]):
                    thisx = [X1[k,i,j], X2[k,i,j]]
                    thisy = [Y1,Y2[k,i,j]]
                    LINE[sub2ind(dim_sub,i,j)-1].set_data(thisx, thisy)
                    LINE[Nplot+sub2ind(dim_sub,i,j)-1].set_text(time_template % (k * dt))
            return LINE

        return animation.FuncAnimation(fig, animate, np.arange(1, N), interval=dt*1000, blit=True, init_func=init)
    
    def simulate_ES_control(self):
        ## Run a bunch of trials using the energy shaping controller
        # TODO: update this to the method I defined in misc/dimension_test
        
        # parameters for the amount of different trajectories we generate with the energy shaping controller
        num_trials = 1
        num_states = 4
        num_t = len(self.time)
        y = np.zeros((num_t, num_states, num_trials))
        
        for i in range(num_trials):
            # initial conditions
            theta = 0
            x = 0.0
            th_dot = 2*(i/num_trials) - 1 
            xdot = 0.0
        
            # initial state
            self.init_state = np.array([theta, x, th_dot, xdot])
            self.u_hist = []
            # integrate the ODE using scipy.integrate.
            # TODO switch over to the more modern solve_ivp, as we do for the pendubot
            y[:, :, i] = integrate.odeint(self.derivs, self.init_state, self.time)
            u = np.zeros((y.shape[0],1))
            for t in range(len(self.time)):
                    u[t] = self.control(y[t]) 
                
        return y, u
    def expert(self, y):
            u = np.zeros((y.shape[0],1))
            for t in range(len(self.time)):
                    u[t] = self.controlES(y[t])
            return u
    def simulate_NN_control(self, net, netType, Ts = None, look_back = None):

        self.u_hist = []
        self.Ts = Ts
        if(netType == 'FF'):
            # Feed forward 
            self.control = self.make_ff_controller(net)
            # Run the simulation for the feedforward network
            # Fill in our u after the fact..
            y = integrate.odeint(self.derivs_dig, self.init_state, self.time, hmax = self.dt/3) 
        elif(netType == 'FFLB'):
            # Feed forward with looking back
            self.control = self.make_fflb_controller(net, look_back)                
            # Run the simulation for the Feedforward look back network
            # integrate the ODE using scipy.integrate.
            # Fill in our u after the fact..
            self.look_back = look_back
            y = integrate.odeint(self.derivs_dig_lb, self.init_state, self.time, hmax = self.dt/3)
        elif(netType == 'LSTM'):
            # Long short-term memory
            self.control = self.make_lstm_controller(net, look_back)
            # integrate the ODE using scipy.integrate.
            # Fill in our u after the fact..
            self.look_back = look_back
            y = integrate.odeint(self.derivs_dig_lb, self.init_state, self.time,  hmax = self.dt/3)     
        return y, self.u_hist
    
    def make_controller(self, model):
        def nn_controller(q):
            if (q[0] < (140 * (pi/180)) ) or (q[0] > (220 * (pi/180)) ):
                u = model.predict(q.reshape((1,4,1)))
            else:
                # balancing
                # LQR: K values from MATLAB
                k1 = 140.560
                k2 = -3.162
                k3 = 41.772
                k4 = -8.314
                u = -(k1 * (q[0] - pi) + k2 * q[1] + k3 * q[2] + k4 * q[3])
            return min(self.u_max, max(-self.u_max,u))
            
        return nn_controller
    
    
    # an ugly hack TODO make this and the one above compatible
    def make_ff_controller(self, model):
        def nn_controller(q):
            if (q[0] < (140 * (pi/180)) ) or (q[0] > (220 * (pi/180)) ):
                u = model.predict(q.reshape(1,4))
                u =  u[0][0]
            else:
                # balancing
                # LQR: K values from MATLAB
                k1 = 140.560
                k2 = -3.162
                k3 = 41.772
                k4 = -8.314
                u = -(k1 * (q[0] - pi) + k2 * q[1] + k3 * q[2] + k4 * q[3])
            return min(self.u_max, max(-self.u_max,u))
            
        return nn_controller
    
    def make_fflb_controller(self, model, look_back):
        def nn_controller(q):
            if (q[0,look_back-1] < (140 * (pi/180)) ) or (q[0,look_back-1] > (220 * (pi/180)) ):
                u = model.predict(q.reshape(1,q.shape[0],q.shape[1]))
                u = u[0][0]
            else:
                # balancing
                # lqr: k values from matlab
                k1 = 140.560
                k2 = -3.162
                k3 = 41.772
                k4 = -8.314
                u = -(k1 * (q[0,look_back-1] - pi) + k2 * q[1,look_back-1] + k3 * q[2,look_back-1] + k4 * q[3,look_back-1])
            return min(self.u_max, max(-self.u_max,u))
            
        return nn_controller
    
    def make_lstm_controller(self, model, look_back):
        def nn_controller(q):
            if (q[0,look_back-1] < (140 * (pi/180)) ) or (q[0,look_back-1] > (220 * (pi/180)) ):
                q = np.swapaxes(q,0,1)
                u = model.predict(q.reshape(1,q.shape[0],q.shape[1]))
                u = u[0][0]
            else:
                # balancing
                # lqr: k values from matlab
                k1 = 140.560
                k2 = -3.162
                k3 = 41.772
                k4 = -8.314
                u = -(k1 * (q[0,look_back-1] - pi) + k2 * q[1,look_back-1] + k3 * q[2,look_back-1] + k4 * q[3,look_back-1])
            return min(self.u_max, max(-self.u_max,u))
        
        return nn_controller
    
    
    def calc_feat(self, y):

        # remove sign
        y_abs = np.abs(y)
        # remove periodicity of angle
        y_abs[:,0] = np.mod(y_abs[:,0], 2*pi)
        
        err = np.sqrt(np.sum((y_abs-self.final_state)**2,1))
        
        t_err = None
        for i in range(0,len(err)):
            if((err[i]<=self.err_final)&(t_err==None)):
                t_err = self.time[i]
            if(self.err_final<err[i]):
                t_err = None
                
        return t_err
            
            
   # @jit(nopython=False)
    def controlES(self, q):
        """
        This is where you should define the control for the cartpole, called by derivs.

        By default, implements a swingup controller for the cartpole based on energy shaping. Switches to an LQR to
        balance the pendulum

        :param q: numpy array of state variables [theta, x, thetadot, xdot]
        :return: u, the control torque in N*m
        """

        if (q[0] < 140 * pi/180) or (q[0] > 220 * pi/180 ):
            # swing up
            # energy error: Ee
            Ee = 0.5 * self.mp * self.L * self.L * q[2] ** 2 - self.mp * self.g * self.L * (1 + cos(q[0]))
            # energy control gain:
            k = 0.23
            # input acceleration: A (of cart)
            A = k * Ee * cos(q[0]) * q[2]
            # convert A to u (using EOM)
            delta = self.mp * sin(q[0]) ** 2 + self.mc
            u = A * delta - self.mp * self.L * (q[2] ** 2) * sin(q[0]) - self.mp * self.g * sin(q[2]) * cos(q[2])
        else:
            # balancing
            # LQR: K values from MATLAB
            k1 = 140.560
            k2 = -3.162
            k3 = 41.772
            k4 = -8.314
            u = -(k1 * (q[0] - pi) + k2 * q[1] + k3 * q[2] + k4 * q[3])
        return min(self.u_max, max(-self.u_max,u))
            
            
