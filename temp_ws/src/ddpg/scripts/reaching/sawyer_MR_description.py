#-----------------------------------------------------------------------
# Tasks:
# 1. Returns M0 matrix and body twists for Sawyer
#
# Obtained from Chainatee Tanakulrungson
#-----------------------------------------------------------------------

import numpy as np
from math import cos, sin, radians

s10 = sin(radians(10))
c10 = cos(radians(10))

# B_i is the screw axis of the joint expressed in ee frame, when manipulator is @ zero position
# spatial velocity in body frame


# Blist = np.array([[ s10,  -c10,   0.,			-1.0155*c10, 	-1.0155*s10, 	-0.1603],  				#J1
#                   [-c10,  -s10,   0.,			-0.9345*s10, 	0.9345*c10, 		0.],       			#J2
#                   [ 0. ,    0.,   1., 			-0.0322*s10, 	0.0322*c10, 		0.],		  		#J3
#                   [-c10,  -s10,   0., 			-0.5345*s10, 	0.5345*c10, 		0.],       			#J4
#                   [0.,      0.,   1., 			 0.1363*s10, 	-0.1363*c10, 		0.],           		#J5
#                   [-c10, -s10,    0., 			-0.1345*s10,	0.1345*c10, 		0.],       			#J6
#                   [0.,     0.,    1.,		     0., 			0., 				0.]])               #J7


Blist = np.array([[ s10,  -c10,   0.,			-1.1511*c10, 	-1.1511*s10, 	-0.1603],  				#J1, if joint 1 rotates w1 = 1 rad/s
                  [-c10,  -s10,   0.,			-0.9345*s10, 	0.9345*c10, 		0.],       			#J2
                  [ 0. ,    0.,   1., 			-0.0322*s10, 	0.0322*c10, 		0.],		  		#J3
                  [-c10,  -s10,   0., 			-0.5345*s10, 	0.5345*c10, 		0.],       			#J4
                  [0.,      0.,   1., 			 0.1363*s10, 	-0.1363*c10, 		0.],           		#J5
                  [-c10, -s10,    0., 			-0.1345*s10,	0.1345*c10, 		0.],       			#J6
                  [0.,     0.,    1.,		     0., 			0., 				0.]])               #J7

# z-axis of joint 3,5, and 7 are parallel to that of ee frame 

Blist = Blist.T

# M >> ee configurationin SE(3) when manipulator is at zero configuration


# rotation matrix means the
# *** What I should do first.
# 		when executing random pose script, find the rotation matrix and translation matrix and return the 
# 		M from base to EE
# *** What about the body twist matrix? >> refer to the code


# M = np.array([[  0.,   0.,   1., 1.0155],
#               [-c10, -s10,   0., 0.1603],
#               [ s10, -c10,   0., 0.317],
#               [  0.,   0.,   0., 1.]])

# M = np.array([[  0.,   0.,   1., 1.0155],
#               [-c10, -s10,   0., 0.1603],
#               [ s10, -c10,   0., 0.317],
#               [  0.,   0.,   0., 1.]])

M = np.array([[  0.,   0.,   1., 1.1511], # translation should be calibrated with the gripper
              [-c10, -s10,   0., 0.1603],
              [ s10, -c10,   0., 0.317],
              [  0.,   0.,   0., 1.]])



# translation : acquired from 'rqt's /robot/limb/right/endpoint_state' when
# when all joint angles are set to zero


# deg 10 means the e.e's rotation angle 
