#!/usr/bin/env python
#-----------------------------------------------------------------------
# Stripped down version of velocity_controller.py
# Runs at 100Hz
#
# Tasks:
# 1. Finds Jb and Vb
# 2. Uses naive least-squares damping to find qdot
#   (Addresses inf vel at singularities)
# 3. Publishes qdot to Sawyer using the limb interface
#
# Written By Stephanie L. Chang
# Last Updated: 4/13/17
#-----------------------------------------------------------------------
# Python Imports
import numpy as np
from math import pow, pi, sqrt
import tf.transformations as tr
import threading
import tf

# ROS Imports
import rospy
from std_msgs.msg import Bool, Int32, Float64
from geometry_msgs.msg import Pose, Point, Quaternion
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import intera_interface

# Local Imports
import sawyer_MR_description as s
import modern_robotics as r
import custom_logging as cl

####################
# GLOBAL VARIABLES #
####################
TIME_LIMIT = 7    #15s
DAMPING = 0.06
JOINT_VEL_LIMIT = 2    #2rad/s

class VelocityControl(object):
    def __init__(self):
        rospy.loginfo("Creating VelocityController class")

        # Create KDL model
        with cl.suppress_stdout_stderr():    # Eliminates urdf tag warnings
            self.robot = URDF.from_parameter_server()
        # self.kin = KDLKinematics(self.robot, "base", "right_gripper")
        self.kin = KDLKinematics(self.robot, "base", "right_gripper_tip")
        self.names = self.kin.get_joint_names()

        # Limb interface
        self.arm = intera_interface.Limb("right")
        self.hand = intera_interface.gripper.Gripper('right')

        # Grab M0 and Blist from saywer_MR_description.py
        self.M0 = s.M #Zero config of right_hand
        self.Blist = s.Blist #6x7 screw axes mx of right arm
        self.Kp = 10*np.eye(6)
        self.Ki = 0*np.eye(6)
        self.it_count = 0
        self.int_err = 0


        self.main_js = JointState()
        self.main_js.name = ['right_j0', 'head_pan', 'right_j1', 'right_j2', \
                            'right_j3', 'right_j4', 'right_j5', 'right_j6']

        ##### If I'm sending position messages is this velocity control?
        self.main_js.position = np.ndarray.tolist(np.zeros(8)) #init main_js

        self.it_count = 0
        self.int_err = 0
        # Shared variables
        self.mutex = threading.Lock()
        self.time_limit = rospy.get_param("~time_limit", TIME_LIMIT)
        self.damping = rospy.get_param("~damping", DAMPING)
        self.joint_vel_limit = rospy.get_param("~joint_vel_limit", JOINT_VEL_LIMIT)
        self.q = np.zeros(7)        # Joint angles
        self.qdot = np.zeros(7)     # Joint velocities
        self.T_goal = np.array(self.kin.forward(self.q))    # Ref se3
        # print self.T_goal
        self.isCalcOK = False
        # Subscriber
        self.ref_pose_sub = rospy.Subscriber('/teacher/ik_vel/', Pose, self.ref_pose_cb)
        # self.ref_pose_sub = rospy.Subscriber('/ddpg/vel_start/', Float64, self.vel_start_cb)
        # self.ref_pose_sub = rospy.Subscriber('/ddpg/vel_end/', Float64, self.vel_end_cb)
        # self.target_pose_calc_ = rospy.Subscriber('/teacher/ik_vel/', Pose, self.ref_pose_cb)
        # Command publisher
        # self.vel_command_list_pub = rospy.Publisher('/vel/command/', Pose, queue_size = 3)

        self.hand.calibrate()

        # # rospy.wait_for_message("/ddpg/reset/",Float64)
        self.r = rospy.Rate(100)
        while not rospy.is_shutdown():
            if rospy.has_param('vel_calc'):
                # print 'calc vel'
                self.calc_joint_vel()
                self.r.sleep()
            else:
                # print '...'
                pass

    # def vel_start_cb(self, start):
    #     print 'OK to caculate&command joint vel'
    #     self.isCalcOK = True

    # def vel_end_cb(self, end):
    #     print 'Vel calculation has finished'
    #     self.isCalcOK = False


    def ref_pose_cb(self, some_pose): # Takes target pose, returns ref se3
        rospy.logdebug("ref_pose_cb called in velocity_control.py")
        # p = np.array([some_pose.position.x, some_pose.position.y, some_pose.position.z])
        p = np.array([some_pose.position.x, some_pose.position.y, some_pose.position.z])
        # print p
        quat = np.array([some_pose.orientation.x, some_pose.orientation.y, some_pose.orientation.z, some_pose.orientation.w])
        goal_tmp = tr.compose_matrix(angles=tr.euler_from_quaternion(quat, 'sxyz'), translate=p) # frame is spatial 'sxyz', return Euler angle from quaternion for specified axis sequence.
        with self.mutex:
            self.T_goal = goal_tmp
        # self.joint_com_published = True

    def get_q_now(self):         # Finds current joint angles
        qtemp = np.zeros(7)
        i = 0
        while i<7:
            qtemp[i] = self.arm.joint_angle(self.names[i])
            i += 1
        with self.mutex:
            self.q = qtemp              # Angles in radians

    def stop_oscillating(self):
        i = 0
        v_norm = 0
        qvel = self.qdot

        while i<7:
            v_norm += pow(qvel[i],2)
            i += 1
        v_norm = sqrt(v_norm)

        if v_norm < 0.1:
            self.qdot = np.zeros(7)
        return

    def calc_joint_vel(self):

        rospy.logdebug("Calculating joint velocities...")

        # Body stuff
        Tbs = self.M0 # 
        Blist = self.Blist # 

        # Current joint angles
        self.get_q_now()
        with self.mutex:
            q_now = self.q #1x7 mx

        # Desired config: base to desired - Tbd
        with self.mutex:
            T_sd = self.T_goal # 

        # Find transform from current pose to desired pose, error
        # refer to CH04 >> the product of exponential formula for open-chain manipulator
        # sequence:
        #   1. FKinBody (M, Blist, thetalist)
        #       M: the home config. of the e.e.
        #       Blist: the joint screw axes in the ee frame, when the manipulator is @ the home pose
        #       thetalist : a list of current joints list
        #       We get the new transformation matrix T for newly given joint angles
        #   
        #           1). np.array(Blist)[:,i] >> retrieve one axis' joint screw 
        #           2). ex)  [s10, -c10, 0., -1.0155*c10, -1.0155*s10, -0.1603] -> S(theta)
        #           3). _out = VecTose3(_in)
        #                 # Takes a 6-vector (representing a spatial velocity).
        #                 # Returns the corresponding 4x4 se(3) matrix.
        #           4). _out = MatrixExp6(_in)
        #                    # Takes a se(3) representation of exponential coordinate 
        #                    # Returns a T matrix SE(3) that is achieved by traveling along/about the
        #                    # screw axis S for a distance theta from an initial configuration T = I(dentitiy)
        #           5). np.dot (M (from base to robot's e.e. @home pose , and multiplying exp coord(6), we get new FK pose
        #
        #   2. TransInv >> we get the inverse of homogen TF matrix
        #   3. error = np.dot (T_new, T_sd) >> TF matrix from cur pose to desired pose
        #   4. Vb >>compute the desired body twist vector from the TF matrix 
        #   5. JacobianBody:
        #           # In: Blist, and q_now >> Out : T IN SE(3) representing the end-effector frame when the joints are
        #                 at the specified coordinates

        e = np.dot(r.TransInv(r.FKinBody(Tbs, Blist, q_now)), T_sd) # Modern robotics pp 230 22Nff

        # Desired TWIST: MatrixLog6 SE(3) -> se(3) exp coord
        Vb = r.se3ToVec(r.MatrixLog6(e))
        # Construct BODY JACOBIAN for current config
        Jb = r.JacobianBody(Blist, q_now) #6x7 mx
        # WE NEED POSITION FEEDBACK CONTROLLER

        # Desired ang vel - Eq 5 from Chiaverini & Siciliano, 1994
        # Managing singularities: naive least-squares damping
        n = Jb.shape[-1] #Size of last row, n = 7 joints
        # OR WE CAN USE NUMPY' PINV METHOD 


        invterm = np.linalg.inv(np.dot(Jb.T, Jb) + pow(self.damping, 2)*np.eye(n)) # 
        qdot_new = np.dot(np.dot(invterm,Jb.T),Vb) # It seems little bit akward...? >>Eq 6.7 on pp 233 of MR book

        # Scaling joint velocity
        # minus_v = abs(np.amin(qdot_new))
        # plus_v = abs(np.amax(qdot_new))
        # if minus_v > plus_v:
        #     scale = minus_v
        # else:
        #     scale = plus_v
        # if scale > self.joint_vel_limit:
        #     # qdot_new = 2.0*(qdot_new/scale)*self.joint_vel_limit
        #     qdot_new = 1.0*(qdot_new/scale)*self.joint_vel_limit
        self.qdot = qdot_new #1x7

        # Constructing dictionary
        qdot_output = dict(zip(self.names, self.qdot))

        # Setting Sawyer right arm joint velocities
        self.arm.set_joint_velocities(qdot_output)
        # print qdot_output
        return

    # def calc_joint_vel_2(self):
    #     # self.it_count += 1
    #     # print self.it_count
    #     self.cur_theta_list = np.delete(self.main_js.position, 1)
    #     self.X_d = self.tf_lis.fromTranslationRotation(self.p_d, self.Q_d)
    #     self.X = mr.FKinBody(self.M, self.B_list, self.cur_theta_list)
    #     self.X_e = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(self.X), self.X_d)))
    #     if np.linalg.norm(self.int_err) < 10 and np.linalg.norm(self.int_err) > -10:
    #         self.int_err = (self.int_err + self.X_e)
    #     self.V_b = np.dot(self.Kp, self.X_e) + np.dot(self.Ki, self.int_err)
    #     self.J_b = mr.JacobianBody(self.B_list, self.cur_theta_list)
    #     T_selector = np.matrix([0,0,0],\
    #                            [0,0,0],\
    #                            [1,0,0],\
    #                            [0,1,0],\
    #                            [0,0,1],\
    #                            [0,0,0])
    #     self.theta_dot = np.dot(T_selector, np.dot(np.linalg.pinv(self.J_b), self.V_b))
    #     self.delt_theta = self.theta_dot*(1.0/20)
    #     self.delt_theta = np.insert(self.delt_theta, 1, 0)
    #     self.main_js.position += self.delt_theta
    #     # self.mean_err_msg.data = np.mean(self.X_e)
    #     # print self.mean_err_msg.data
    #     print np.linalg.norm(self.int_err)
def main():
    rospy.init_node('velocity_control')

    try:
        vc = VelocityControl()
        # rospy.wait_for_message("/ddpg/reset/",Float64)
        # r = rospy.Rate(100)
        # while not rospy.is_shutdown():
        #     vc.calc_joint_vel()
        #     r.sleep()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()

if __name__ == '__main__':
    main()
