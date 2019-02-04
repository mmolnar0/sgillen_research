import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    def __init__(self):
        self.L = 1.0;  # length of the pole (m)
        self.mc = 4.0  # mass of the cart (kg)
        self.mp = 1.0  # mass of the ball at the end of the pole

        self.g = 9.8;


    # TODO, refacter to switch t,y -> y,t to be consitent with the derivs
    def animate_cart(self, t, y):
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

        dt = (t[-1] - t[0])/len(t)


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
            time_text.set_text(time_template % (i * dt))
            return line, time_text

        return animation.FuncAnimation(fig, animate, np.arange(1, len(y)), interval=40, blit=True, init_func=init)

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
        return u

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

        dqdt = np.zeros_like(q)

        # control input
        u = self.control(q)

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
