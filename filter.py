# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
#import params

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass
    
#     def __init__(self):
#         self.dim_state = 4  #dim_state from param.py  # process model dimension
#         self.dt = 0.1 #from param.py  # time increment
#         self.q=0.1 #from param.py  # process noise variable for Kalman filter Q

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        dt = params.dt
        return np.matrix([[1, 0, 0, dt, 0, 0],
                          [0, 1, 0, 0, dt, 0],
                          [0, 0, 1, 0, 0, dt],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])

        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############

        q = params.q
        dt = params.dt
        q3 = ((dt**3)/3)*  q 
        q2 = ((dt**2)/2)*  q 
        q1 = dt * q 
        return np.matrix([[q3, 0 , 0 , q2, 0 , 0 ],
                          [0 , q3, 0 , 0 , q2, 0 ],
                          [0 , 0 , q3, 0 , 0 , q2],
                          [q2, 0 , 0 , q1, 0 , 0 ],
                          [0 , q2, 0 , 0 , q1, 0 ],
                          [0 , 0 , q2, 0 , 0 , q1]])
        
        ############
        # END student code
        ############ 

#     def H(self): #used meas.sensor.get_H() instead
#         # measurement matrix H
#         return np.matrix([[1, 0, 0, 0],
#                        [0, 1, 0, 0]]) 
    
    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        F = self.F()
        Q = self.Q()
        x = track.x
        P = track.P
        x_predicted = F * x
        P_predicted = F * P * F.T + Q
        
        track.set_x(x_predicted)
        track.set_P(P_predicted)
        #x = track.x
#         P = track.P
#         F = self.F()
#         x_predected = F*track.x # state prediction
#         P = F*P*F.transpose() + self.Q() # covariance prediction
#         return x, P
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        #parameters
        x = track.x
        R = meas.R #measurement noise
#         meas.z #measurement
        P = track.P
        H = meas.sensor.get_H(x) # measurement matrix
        gamma = self.gamma(track, meas) # residual
        S = self.S(track, meas,H) # covariance of residual
        
        #equations
        K = P*H.transpose()*np.linalg.inv(S) # Kalman gain
        x_updated = x + K*gamma # state update
        I = np.identity(params.dim_state)
        P_updated = (I - K*H)*  P # covariance update
        #return x, P     
        #track.append([x,P])
        track.set_x(x_updated)
        track.set_P(P_updated)
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        x = track.x
        H_x = meas.sensor.get_hx(x) # measurement matrix
        
        z = meas.z #measurement
        gamma = z - H_x # residual
        return gamma
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        R = meas.R #measurement noise
        #H = self.H() # measurement matrix
        P = track.P
        S = H*P*H.transpose() + R # covariance of residual
        return S
        
        ############
        # END student code
        ############ 