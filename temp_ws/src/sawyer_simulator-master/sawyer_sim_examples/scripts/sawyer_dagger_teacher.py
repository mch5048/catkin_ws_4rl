#!/usr/bin/env python

# Copyright (c) 2015-2018, Rethink Robotics Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sawyer SDK Inverse Kinematics Pick and Place Demo
"""
import argparse
import struct
import sys
import copy

import rospy
import rospkg

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

from random import *
import intera_interface
from sensor_msgs.msg import JointState, Image
from intera_core_msgs.msg import JointCommand



from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from cv_bridge import CvBridge
# from berkeley_sawyer.srv import *
from intera_motion_interface import motion_waypoint
from intera_motion_interface import motion_trajectory
from std_msgs.msg import *
from std_srvs.srv import *
import math
from math import degrees as deg
from math import radians as rad

from tf_conversions import posemath
from tf.msg import tfMessage
from tf.transformations import quaternion_from_euler
import PyKDL
from intera_interface import Limb
from intera_interface import Cuff
from intera_interface import CHECK_VERSION
from tf_conversions import posemath
from tf.msg import tfMessage
from tf.transformations import quaternion_from_euler
from intera_core_msgs.msg import (
    DigitalIOState,
    DigitalOutputCommand,
    IODeviceStatus
)

from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
    SolvePositionIK,
    SolvePositionIKRequest,
)

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header
from intera_motion_msgs.msg import (
    Trajectory,
    TrajectoryOptions,
    Waypoint
)
from intera_motion_interface import (

    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)

import math
from math import pi
import numpy as np

import os
base_dir = os.path.dirname(os.path.realpath(__file__))


print base_dir

# class trajectorySender(object):

#     def __init__(self, parent = None):

#         def __init__(self, limb="right", hover_distance = 0.15, tip_name="right_gripper_tip"):
#         self._limb_name = limb # string
#         self._tip_name = tip_name # string
#         self._hover_distance = hover_distance # in meters
#         self._limb = intera_interface.Limb(limb)
#         self._gripper = intera_interface.Gripper()
#         # verify robot is enabled
#         print("Getting robot state... ")
#         self._rs = intera_interface.RobotEnable(intera_interface.CHECK_VERSION)
#         self._init_state = self._rs.state().enabled
#         print("Enabling robot... ")
#         self._rs.enable()

#         self.trajectory_option = TrajectoryOptions(interpolation_type='CARTESIAN')
#         self.trajectory = MotionTrajectory(trajectory_options=self.trajectory_option, limb=self.limb)
#         self.temp_trajectory = MotionTrajectory(trajectory_options=self.trajectory_option, limb=self.limb)

#         self.wpt_opts = MotionWaypointOptions(max_linear_speed=0.3,
#                                               max_linear_accel=0.3,
#                                               max_rotational_speed=3.5,
#                                               max_rotational_accel=3.5,
#                                               max_joint_speed_ratio=0.7,
#                                               corner_distance=0.3)

#         self.waypoint = MotionWaypoint(options=self.wpt_opts.to_msg(), limb=self.limb)
#         self.waypoint_initial = MotionWaypoint(options=self.wpt_opts.to_msg(), limb=self.limb)
#         self.temp_waypoint = MotionWaypoint(options=self.wpt_opts.to_msg(), limb=self.limb)


class PickAndPlace(object):
    def __init__(self, limb="right", hover_distance = 0.15, tip_name="right_gripper_tip"):
        self._limb = Limb()
        self._limb.set_joint_position_speed(0.1)

        self._tip_name = tip_name # string
        self._hover_distance = hover_distance # in meters
        self._gripper = intera_interface.Gripper()
        self.head_display = intera_interface.HeadDisplay()
        self.head_display.display_image(base_dir + "/head.png")
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(intera_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

        self.joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']

        self.joint_dict = {}
        self.vel_ik_pos_pub = rospy.Publisher('/teacher/ik_vel/', Pose, queue_size = 3)


        # self.trajectory_option = TrajectoryOptions(interpolation_type='CARTESIAN')
        # self.trajectory = MotionTrajectory(trajectory_options=self.trajectory_option, limb=self.limb)
        # self.temp_trajectory = MotionTrajectory(trajectory_options=self.trajectory_option, limb=self.limb)

        # self.wpt_opts = MotionWaypointOptions(max_linear_speed=0.3,
        #                                       max_linear_accel=0.3,
        #                                       max_rotational_speed=3.5,
        #                                       max_rotational_accel=3.5,
        #                                       max_joint_speed_ratio=0.7,
        #                                       corner_distance=0.3)

        # self.waypoint = MotionWaypoint(options=self.wpt_opts.to_msg(), limb=self.limb)
        # self.waypoint_initial = MotionWaypoint(options=self.wpt_opts.to_msg(), limb=self.limb)
        # self.temp_waypoint = MotionWaypoint(options=self.wpt_opts.to_msg(), limb=self.limb)

        # self.trajectory_option = TrajectoryOptions(interpolation_type='CARTESIAN')
        # self.trajectory = MotionTrajectory(trajectory_options=self.trajectory_option, limb=self._limb)
        # self.temp_trajectory = MotionTrajectory(trajectory_options=self.trajectory_option, limb=self._limb)

        # self.wpt_opts = MotionWaypointOptions(max_linear_speed=0.3,
        #                                       max_linear_accel=0.3,
        #                                       max_rotational_speed=3.5,
        #                                       max_rotational_accel=3.5,
        #                                       max_joint_speed_ratio=0.7,
        #                                       corner_distance=0.3)

        # self.waypoint = MotionWaypoint(options=self.wpt_opts.to_msg(), limb=self._limb)
        # self.waypoint_initial = MotionWaypoint(options=self.wpt_opts.to_msg(), limb=self._limb)
        # self.temp_waypoint = MotionWaypoint(options=self.wpt_opts.to_msg(), limb=self._limb)

    def set_joint_velocities(self, velocities):
            """
            Commands the joints of this limb to the specified velocities.
            B{IMPORTANT}: set_joint_velocities must be commanded at a rate great
            than the timeout specified by set_command_timeout. If the timeout is
            exceeded before a new set_joint_velocities command is received, the
            controller will switch modes back to position mode for safety purposes.
            @type velocities: dict({str:float})
            @param velocities: joint_name:velocity command
            """
            self._command_msg.names = velocities.keys()
            self._command_msg.velocity = velocities.values()
            self._command_msg.mode = JointCommand.VELOCITY_MODE
            self._command_msg.header.stamp = rospy.Time.now()
            self._pub_joint_cmd.publish(self._command_msg)







    def addCurrentPoseWayPoint(self):
        pos, orit = self.getCurrentPose()

        self.quat_angle = rot_command.GetQuaternion()
        self.targetPose.position.x = pos[0]
        self.targetPose.position.y = pos[1]
        self.targetPose.position.z = pos[2]
        self.targetPose.orientation.x = orit[0]
        self.targetPose.orientation.y = orit[1]
        self.targetPose.orientation.z = orit[2]
        self.targetPose.orientation.w = orit[3]
        self.poseStamped.pose = self.targetPose
        self.waypoint.set_cartesian_pose(self.poseStamped, 'right_hand')
        self.trajectory.append_waypoint(self.waypoint.to_msg())

    def addMotionWayPoints(self, x, y, z, roll=-179.0, pitch=0.0, yaw=100.0, x_offset=0.0, y_offset=0.0, z_offset=0.0):
        rot_command = PyKDL.Rotation.RPY(rad(roll), rad(pitch), rad(yaw))
        self.quat_angle = rot_command.GetQuaternion()
        self.targetPose.position.x = x / 1000.0 + x_offset
        self.targetPose.position.y = y / 1000.0 + y_offset
        self.targetPose.position.z = z / 1000.0 + z_offset + self.gripperlength
        self.targetPose.orientation.x = self.quat_angle[0]
        self.targetPose.orientation.y = self.quat_angle[1]
        self.targetPose.orientation.z = self.quat_angle[2]
        self.targetPose.orientation.w = self.quat_angle[3]
        self.poseStamped.pose = self.targetPose
        # print (self.poseStamped.pose)
        self.waypoint.set_cartesian_pose(self.poseStamped, 'right_hand', None)
        self.trajectory.append_waypoint(self.waypoint.to_msg())


    def sendTrajectory(self):
        log = self.trajectory.send_trajectory(timeout=10)
        print (log)
        if self._limb.has_collided():
            rospy.logerr('collision detected!!!')
            rospy.sleep(.5)
        self.clearTrajectory()

    def clearTrajectory(self):
        self.trajectory.clear_waypoints()




    def getCurrentPose(self):

        self.endpoint_pose = self._limb.endpoint_pose()
        self.endpoint_position = self.endpoint_pose['position']
        self.endpoint_orientation = self.endpoint_pose['orientation']

        return self.endpoint_position, self.endpoint_rotation


    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format('right'))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        self.gripper_open()

    def _guarded_move_to_joint_position(self, joint_angles, timeout=5.0):
        if rospy.is_shutdown():
            return
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles,timeout=timeout)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def gripper_open(self):
        self._gripper.open()
        rospy.sleep(1.0)

    def gripper_close(self):
        self._gripper.close()
        rospy.sleep(1.0)

    def _approach(self, pose):
        approach = copy.deepcopy(pose)
        # approach with a pose the hover-distance above the requested pose
        approach.position.z = approach.position.z + self._hover_distance
        joint_angles = self._limb.ik_request(approach, self._tip_name)
        self._guarded_move_to_joint_position(joint_angles)

    def _retract(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z + self._hover_distance
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        self._servo_to_pose(ik_pose)

    def _servo_to_pose(self, pose, time=4.0, steps=400.0):
        ''' An *incredibly simple* linearly-interpolated Cartesian move '''
        # r = rospy.Rate(1/(time/steps)) # Defaults to 100Hz command rate
        r = rospy.Rate(100.0) # Defaults to 100Hz command rate
        current_pose = self._limb.endpoint_pose()
        ik_delta = Pose()
        ik_delta.position.x = (current_pose['position'].x - pose.position.x) / steps
        ik_delta.position.y = (current_pose['position'].y - pose.position.y) / steps
        ik_delta.position.z = (current_pose['position'].z - pose.position.z) / steps
        ik_delta.orientation.x = (current_pose['orientation'].x - pose.orientation.x) / steps
        ik_delta.orientation.y = (current_pose['orientation'].y - pose.orientation.y) / steps
        ik_delta.orientation.z = (current_pose['orientation'].z - pose.orientation.z) / steps
        ik_delta.orientation.w = (current_pose['orientation'].w - pose.orientation.w) / steps
        for d in range(int(steps), -1, -1):
            if rospy.is_shutdown():
                return
            ik_step = Pose()
            ik_step.position.x = d*ik_delta.position.x + pose.position.x
            ik_step.position.y = d*ik_delta.position.y + pose.position.y
            ik_step.position.z = d*ik_delta.position.z + pose.position.z
            ik_step.orientation.x = d*ik_delta.orientation.x + pose.orientation.x
            ik_step.orientation.y = d*ik_delta.orientation.y + pose.orientation.y
            ik_step.orientation.z = d*ik_delta.orientation.z + pose.orientation.z
            ik_step.orientation.w = d*ik_delta.orientation.w + pose.orientation.w
            joint_angles = self._limb.ik_request(ik_step, self._tip_name)
            if joint_angles:
                self._limb.set_joint_positions(joint_angles)
            else:
                rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")
            r.sleep()
        rospy.sleep(1.0)


    def reach(self, pose):
        if rospy.is_shutdown():
            return
        # open the gripper
        # servo above pose
        # self._approach(pose)
        # servo to pose

        time = uniform(7.0,10.0)
        steps = time *100.0
        self._servo_to_pose(pose=pose, time=time, steps=steps)
        if rospy.is_shutdown():
            return
        # close gripper
        # retract to clear object

    def fk_service_client(self, joint_pose, limb = "right"):
        # returned value contains PoseStanped
        # std_msgs/Header header
        # geometry_msgs/Pose pose
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/FKService"
        fksvc = rospy.ServiceProxy(ns, SolvePositionFK)
        fkreq = SolvePositionFKRequest()
        joints = JointState()
        joints.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
                         'right_j4', 'right_j5', 'right_j6']
          # joints.position = [0.763331, 0.415979, -1.728629, 1.482985,
          #                    -1.135621, -1.674347, -0.496337]
        joints.position = joint_pose
          # Add desired pose for forward kinematics
        fkreq.configuration.append(joints)
          # Request forward kinematics from base to "right_hand" link
        fkreq.tip_names.append('right_hand')
          # fkreq.tip_names.append('right_gripper_tip')

        try:
            rospy.wait_for_service(ns, 5.0)
            resp = fksvc(fkreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False

          # Check if result valid
        if (resp.isValid[0]):
            rospy.loginfo("SUCCESS - Valid Cartesian Solution Found")
            rospy.loginfo("\nFK Cartesian Solution:\n")
            rospy.loginfo("------------------")
            rospy.loginfo("Response Message:\n%s", resp)
              # print resp.pose_stamp[0].pose
            return resp.pose_stamp[0].pose
        else:
            rospy.logerr("INVALID JOINTS - No Cartesian Solution Found.")
            return False


    def reach_vel_ctrl(self, pose):
        '''
            using IK service, reach target position
        '''  
        target_pose = pose
        joint_angles = self._limb.ik_request(target_pose, self._tip_name)
        print '===================='
        print joint_angles
        print '===================='

        target_pose = [joint_angles['right_j0'], joint_angles['right_j1'],
        joint_angles['right_j2'], joint_angles['right_j3'],
        joint_angles['right_j4'], joint_angles['right_j5'],
        joint_angles['right_j6']]
        _pose = self.fk_service_client(target_pose)
        if _pose: # if Forward Kinematics solution exists
            r = rospy.Rate(100) # command for 30Hz
            while not rospy.is_shutdown():
                self.vel_ik_pos_pub.publish(_pose)
                _current_pose = self._limb.endpoint_pose()
                current_pose = np.array((_current_pose['position'].x, _current_pose['position'].y))
                target_pose = np.array(( _pose.position.x, _pose.position.y))
                _err = np.linalg.norm(current_pose-target_pose)
                # print _err
                _err_2 = (_pose.position.z - _current_pose['position'].z)

                if _err <= 0.05 and _err_2<=0.17: # if reached for target object in 5cm
                    print 'Reached target object'
                    return True
                    break
                r.sleep() # 
        return False



    def move_to_start_vel_command(self, joint_pose):
        
        _joint_pose = joint_pose
        '''
          1. execute FK (joint pose -> ee pose) for start pose of robot
          2. get Cartesian pose of the robot
          3. 
        '''
        # print _joint_pose
        _pose = self.fk_service_client(_joint_pose)

        if _pose: # if Forward Kinematics solution exists
            r = rospy.Rate(100) # command for 30Hz
            while not rospy.is_shutdown():
                self.vel_ik_pos_pub.publish(_pose)
                _current_pose = self._limb.endpoint_pose()
                current_pose = np.array((_current_pose['position'].x, _current_pose['position'].y))
                target_pose = np.array(( _pose.position.x, _pose.position.y))
                _err = np.linalg.norm(current_pose-target_pose)
                # print _err
                _err_2 = (_pose.position.z - _current_pose['position'].z)

                if _err <= 0.05 and _err_2<=0.17: # if reached for target object in 5cm
                    print 'Reached random start pose'
                    return True
                    break
                r.sleep() # 
        return False



def load_gazebo_models(table_pose=Pose(position=Point(x=0.75, y=0.0, z=0.0)),
                       table_reference_frame="world",
                       block_pose=Pose(position=Point(x=0.4225, y=0.1265, z=1.1725)),
                       block_reference_frame="world",
                       kinect_pose=Pose(position=Point(x=1.50, y=0.0, z=1.50)),
                       kinect_reference_frame="world"
                       ):

    kinect_RPY = PyKDL.Rotation.RPY(0.0, 0.7854, 3.142)
    kinect_quat = kinect_RPY.GetQuaternion()

    kinect_pose.position.x = 1.50
    kinect_pose.position.y = 0.0
    kinect_pose.position.z = 1.50
    kinect_pose.orientation.x = kinect_quat[0]
    kinect_pose.orientation.y = kinect_quat[1]
    kinect_pose.orientation.z = kinect_quat[2]
    kinect_pose.orientation.w = kinect_quat[3]

    # Get Models' Path
    model_path = rospkg.RosPack().get_path('sawyer_sim_examples')+"/models/"
    # Load Table SDF
    table_xml = ''
    with open (model_path + "cafe_table/model.sdf", "r") as table_file:
        table_xml=table_file.read().replace('\n', '')
    #Load kinect SDF
    kinect_xml = ''
    with open (model_path + "kinect/model.sdf", "r") as kinect_file:
        kinect_xml=kinect_file.read().replace('\n', '')
    # Load Block URDF
    block_xml = ''
    with open (model_path + "block/model_1.urdf", "r") as block_file:
        block_xml=block_file.read().replace('\n', '')
    # Spawn Table SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf1 = spawn_sdf("cafe_table", table_xml, "/",
                             table_pose, table_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))


#    try:
#        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
#        resp_sdf2 = spawn_sdf("kinect", kinect_xml, "/",
#                             kinect_pose, kinect_reference_frame)



 #   except rospy.ServiceException, e:
 #       rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Block URDF
    # rospy.wait_for_service('/gazebo/spawn_urdf_model')
    # try:
    #     spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
    #     resp_urdf = spawn_urdf("block", block_xml, "/",
    #                            block_pose, block_reference_frame)
    # except rospy.ServiceException, e:
    #     rospy.logerr("Spawn URDF service call failed: {0}".format(e))

def delete_gazebo_models():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("cafe_table")
    except rospy.ServiceException, e:
        print("Delete Model service call failed: {0}".format(e))

def load_gazebo_block(block_pose=Pose(position=Point(x=0.6725, y=0.1265, z=0.7825)),
                       block_reference_frame="world"):
    # Get Models' Path
    model_path = rospkg.RosPack().get_path('sawyer_sim_examples')+"/models/"
    
    # Load Block URDF
    block_xml = ''

    num = randint(1,3)
    num = str(num)
    with open (model_path + "block/model_"+num+".urdf", "r") as block_file:
        block_xml=block_file.read().replace('\n', '')
    
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf("block", block_xml, "/",
                               block_pose, block_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))

def delete_gazebo_block():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("block")
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))




def load_kinect_camera(kinect_pose=Pose(position=Point(x=1.5, y=0.0, z=1.5)),
                       kinect_reference_frame="world"):


    rand_roll = np.random.uniform(-0.02,0.02)
    rand_pitch = np.random.uniform(pi/3.5, pi/3.54)
    rand_yaw = np.random.uniform(pi - 0.02*pi, pi+ 0.02*pi)


    # kinect_RPY = PyKDL.Rotation.RPY(0.0, pi/2.7, pi)
    kinect_RPY = PyKDL.Rotation.RPY(rand_roll, rand_pitch, rand_yaw)
    kinect_quat = kinect_RPY.GetQuaternion()



    rand_x = np.random.uniform(1.03,1.05)
    rand_y = np.random.uniform(-0.03,0.03)
    # rand_z = uniform(2.2,2.25)
    rand_z = np.random.uniform(1.95,1.96)

    kinect_pose = Pose()

    kinect_pose.position.x = rand_x
    kinect_pose.position.y = rand_y
    kinect_pose.position.z = rand_z
    kinect_pose.orientation.x = kinect_quat[0]
    kinect_pose.orientation.y = kinect_quat[1]
    kinect_pose.orientation.z = kinect_quat[2]
    kinect_pose.orientation.w = kinect_quat[3]



    # Get Models' Path
    model_path = rospkg.RosPack().get_path('sawyer_sim_examples')+"/models/"
    
    # Load Kinect URDF
    kinect_xml = ''
    with open (model_path + "kinect/model.sdf", "r") as kinect_file:
        kinect_xml=kinect_file.read().replace('\n', '')

    # apply randomized pose for domain randimisation
    try:
        spawn_sdf2 = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf2 = spawn_sdf2("kinect", kinect_xml, "/",
                            kinect_pose, kinect_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))
    # Spawn Block URDF


def delete_kinect_camera():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("kinect")
    except rospy.ServiceException, e:
        rospy.loginfo("Delete Model service call failed: {0}".format(e))




def main():
    """SDK Inverse Kinematics Pick and Place Example

    A Pick and Place example using the Rethink Inverse Kinematics
    Service which returns the joint angles a requested Cartesian Pose.
    This ROS Service client is used to request both pick and place
    poses in the /base frame of the robot.

    Note: This is a highly scripted and tuned demo. The object location
    is "known" and movement is done completely open loop. It is expected
    behavior that Sawyer will eventually mis-pick or drop the block. You
    can improve on this demo by adding perception and feedback to close
    the loop.
    """
    np.random.seed(219)
    rospy.init_node("sawyer_dagger_teacher")
    pub_start = rospy.Publisher('/teacher/start', JointCommand, queue_size=1)
    pub_epi_fin = rospy.Publisher('/teacher/fin', JointCommand, queue_size=1)
    vel_ik_pos_pub = rospy.Publisher('/teacher/ik_vel/', Pose, queue_size = 3)
    pub3 = rospy.Publisher('/ddpg/vel_start/', Float64, queue_size=1)
    pub4 = rospy.Publisher('/ddpg/vel_end/', Float64, queue_size=1)
    goal_obs_pub = rospy.Publisher('/teacher/goal_obs/', Pose, queue_size=1)



    rospy.set_param('dagger_reset',"false") # param_name, param_value


    # Load Gazebo Models via Spawning Services
    # Note that the models reference is the /world frame
    # and the IK operates with respect to the /base frame
    # load_gazebo_models()
    # Remove models from the scene on shutdown
    rospy.on_shutdown(delete_gazebo_models)

    limb = 'right'
    hover_distance = 0.15 # meters
    # Starting Joint angles for right arm
    starting_joint_angles = {'right_j0': -0.041662954890248294,
                             'right_j1': -1.0258291091425074,
                             'right_j2': 0.0293680414401436,
                             'right_j3': 1.37518162913313,
                             'right_j4':  -0.06703022873354225,
                             'right_j5': 0.7968371433926965,
                             'right_j6': 1.7659649178699421}

    pnp = PickAndPlace(limb, hover_distance)

    pnp.move_to_start(starting_joint_angles)

    
    # m_planner = trajectorySender()
    # An orientation for gripper fingers to be overhead and parallel to the obj
    overhead_orientation = Quaternion(
                             x=-0.00142460053167,
                             y=0.999994209902,
                             z=-0.00177030764765,
                             w=0.00253311793936)
    block_poses = list()
    # The Pose of the block in its initial location.
    # You may wish to replace these poses with estimates
    # from a perception node.
    block_poses.append(Pose(
        position=Point(x=0.45, y=0.155, z=-0.129),
        orientation=overhead_orientation))
    # Feel free to add additional desired poses for the object.
    # Each additional pose will get its own pick and place.
    block_poses.append(Pose(
        position=Point(x=0.6, y=-0.1, z=-0.129),
        orientation=overhead_orientation))
    # Move to the desired starting angles
    print("Running. Ctrl-c to quit")
    # pnp.move_to_start(starting_joint_angles)
    idx = 0
    rate = rospy.Rate(1)
    block_quat_pose = [0.00142460053167,
                       0.999994209902,
                       0.00177030764765,
                       0.00253311793936]
    if rospy.has_param('vel_calc'):
        rospy.delete_param('vel_calc')
    load_gazebo_models()

    while not rospy.is_shutdown():

        pnp.move_to_start(starting_joint_angles)

        starting_joint_angles['right_j0'] = np.random.uniform(-0.05, 0.05)
        starting_joint_angles['right_j1'] = np.random.uniform(-0.95, -0.85)
        starting_joint_angles['right_j2'] = np.random.uniform(-0.1, 0.1)
        starting_joint_angles['right_j3'] = np.random.uniform(1.6, 1.7)

        # starting_joint_angles['right_j0'] = np.random.uniform(-0.75, 0.75)
        # starting_joint_angles['right_j1'] = np.random.uniform(-0.97, -0.80)
        # starting_joint_angles['right_j2'] = np.random.uniform(-0.15, 0.15)
        # starting_joint_angles['right_j3'] = np.random.uniform(1.55, 1.75)

        start_pose = [starting_joint_angles['right_j0'], starting_joint_angles['right_j1'],
        starting_joint_angles['right_j2'], starting_joint_angles['right_j3'],
        starting_joint_angles['right_j4'], starting_joint_angles['right_j5'],
        starting_joint_angles['right_j6']]

        # demo_pose = [-0.02002539, 
        # 0.822752,
        # -2.0955126, 
        # 2.1725097,
        # 0.70211718, 
        # -1.50036035,
        # -2.204990234]
        
        while not rospy.is_shutdown(): # wait until trajectory is collected for each episode
            if rospy.has_param('dagger_reset'):
                rospy.delete_param('dagger_reset')
                break

        delete_kinect_camera()
        # delete_gazebo_models()
        delete_gazebo_block()
        rand_x = np.random.uniform(0.45, .75)
        rand_y = np.random.uniform(-0.2, 0.33)
        # rand_x = np.random.uniform(0.44,0.68)

        # rand_y = np.random.uniform(-0.20, 0.35)
        pose_block = Pose(position=Point(x=rand_x, y=rand_y, z=1.00)
                        , orientation=overhead_orientation)
        pose_rob = Pose(position=Point(x=rand_x, y=rand_y, z=0.04), orientation=overhead_orientation)                

        rospy.set_param('vel_calc', 'true')
        oktogo = pnp.move_to_start_vel_command(start_pose)
        # oktogo = pnp.move_to_start_vel_command(demo_pose)
        if rospy.has_param('vel_calc'):
            rospy.delete_param('vel_calc')
        # loads env
        load_gazebo_block(block_pose=pose_block)
        load_kinect_camera()

      

        rospy.set_param('vel_calc', 'true')
        print 'Reaching target object... Learning...'
        rospy.set_param('epi_start', 'true')
        reached = pnp.reach_vel_ctrl(pose_rob)
        rospy.sleep(0.5)
        if rospy.has_param('vel_calc'):
            rospy.delete_param('vel_calc')
        # if reached:
        # rospy.set_param('reached', 'true')
        # goal_obs_pub.publish(pose_rob)


        print 'Reached target object! and Goal obs acquired Resetting...'


            
        # rospy.delete_param('demo_success')
                

    return 0

if __name__ == '__main__':
    sys.exit(main())