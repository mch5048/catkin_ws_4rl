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
import os
import argparse
import struct
import sys
import copy
import math
from math import pi
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
from geometry_msgs.msg import Pose, Point
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest
from gazebo_msgs.srv import DeleteModel, DeleteModelRequest
from random import *
from std_msgs.msg import *

import intera_interface
# from sawyer_sim_examples.msg import GoalObs 
from sawyer_sim_examples.msg import GoalObs 

from intera_core_msgs.msg import JointCommand

from string import Template
from tf_conversions import posemath
from tf.msg import tfMessage
from tf.transformations import quaternion_from_euler
import PyKDL

from light_randomizer import LightRandomizer
from shapes_randomizer import ShapesRandomizer

from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
)
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
import numpy as np



base_dir = os.path.dirname(os.path.realpath(__file__))
# SOURCE FOR DOMAIN RANDOMIZATION

# Light Randomizer

_shape_list = {'block': Template("<box><size>$sx $sy $sz</size></box>"),
               'cylinder': Template("<cylinder><radius>$cr</radius><length>$cl</length></cylinder>"),
               'sphere': Template("<sphere><radius>$sr</radius></sphere>")}

_material_list = ['Gazebo/White',
                  'Gazebo/Grey',
                  'Gazebo/Eyes',
                  'Gazebo/FlatBlack',
                  'Gazebo/Black',
                  'Gazebo/Red',
                  'Gazebo/Green',
                  'Gazebo/Yellow',
                  'Gazebo/Purple',
                  'Gazebo/Turquoise',
                  'Gazebo/WhiteEmissive',
                  'Gazebo/RedEmissive',
                  'Gazebo/GreenEmissive',
                  'Gazebo/PurpleEmissive',
                  'Gazebo/BlueLaser',
                  'Gazebo/BlueEmissive',
                  'Gazebo/JointAnchor',
                  'Gazebo/Blue',
                  'Gazebo/Skull',
                  'Gazebo/ExclamationPoint',
                  'Gazebo/QuestionMark',
                  'Gazebo/SmileyHappy',
                  'Gazebo/SmileySad',
                  'Gazebo/SmileyDead',
                  'Gazebo/SmileyPlain',
                  'Gazebo/WoodFloor',
                  'Gazebo/CeilingTiled',
                  'Gazebo/PaintedWall',
                  'Gazebo/PioneerBody',
                  'Gazebo/Pioneer2Body',
                  'Gazebo/Gold',
                  'Gazebo/CloudySky',
                  'Gazebo/RustySteel',
                  'Gazebo/Chrome',
                  'Gazebo/BumpyMetal',
                  'Gazebo/GrayGrid',
                  'Gazebo/Rocky',
                  'Gazebo/GrassFloor',
                  'Gazebo/Rockwall',
                  'Gazebo/RustyBarrel',
                  'Gazebo/WoodPallet',
                  'Gazebo/Fish',
                  'Gazebo/LightWood',
                  'Gazebo/WoodTile',
                  'Gazebo/Brick',
                  'Gazebo/RedTransparent',
                  'Gazebo/GreenTransparent',
                  'Gazebo/BlueTransparent',
                  'Gazebo/DepthMap',
                  'Gazebo/PCBGreen',
                  'Gazebo/Turret',
                  'Gazebo/EpuckBody',
                  'Gazebo/EpuckRing',
                  'Gazebo/EpuckPlate',
                  'Gazebo/EpuckLogo',
                  'Gazebo/EpuckMagenta',
                  'Gazebo/EpuckGold']





# SOURCE FOR DOMAIN RANDOMIZATION


class PickAndPlace(object):
    def __init__(self, limb="right", hover_distance = 0.15, tip_name="right_gripper_tip"):
        self._limb_name = limb # string
        self._tip_name = tip_name # string
        self._hover_distance = hover_distance # in meters
        self._limb = intera_interface.Limb(limb)
        self._gripper = intera_interface.Gripper()
        self.head_display = intera_interface.HeadDisplay()
        self.head_display.display_image(base_dir + "/head.png")
        self.joint_effort = list()
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = intera_interface.RobotEnable(intera_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        self.isReset = False
        # self.pub = rospy.Publisher('/', , queue_size=1)
        rospy.Subscriber('/robot/limb/right/joint_command', JointCommand , self.jointCommandCB)
        rospy.Subscriber('/ddpg/reset2', Bool , self.resetCB)
        self.vel_ik_pos_pub = rospy.Publisher('/teacher/ik_vel/', Pose, queue_size = 3)


    def resetCB(self, isReset):
        if isReset:
            self.isReset = True



    def jointCommandCB(self , cmd):

        self.joint_efforts = [self._limb.joint_effort('right_j0'),
        self._limb.joint_effort('right_j1'),
        self._limb.joint_effort('right_j2'),
        self._limb.joint_effort('right_j3'),
        self._limb.joint_effort('right_j4'),
        self._limb.joint_effort('right_j5'),
        self._limb.joint_effort('right_j6')]

        torque_list = cmd.effort


    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._limb.set_joint_position_speed(0.0001)
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
        self._limb.set_joint_position_speed(0.001)
        self._guarded_move_to_joint_position(joint_angles)
        self._limb.set_joint_position_speed(0.1)

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
        r = rospy.Rate(1/(time/steps)) # Defaults to 100Hz command rate
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

    def pick(self, pose):
        if rospy.is_shutdown():
            return
        # open the gripper
        self.gripper_open()
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        if rospy.is_shutdown():
            return
        # close gripper
        self.gripper_close()
        # retract to clear object
        self._retract()

    def place(self, pose):
        if rospy.is_shutdown():
            return
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        if rospy.is_shutdown():
            return
        # open the gripper
        self.gripper_open()
        # retract to clear object
        self._retract()

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
          # rospy.loginfo("SUCCESS - Valid Cartesian Solution Found")
          # rospy.loginfo("\nFK Cartesian Solution:\n")
          # rospy.loginfo("------------------")
          # rospy.loginfo("Response Message:\n%s", resp)
          # print resp.pose_stamp[0].pose
          return resp.pose_stamp[0].pose
      else:
          rospy.logerr("INVALID JOINTS - No Cartesian Solution Found.")
          return False

    def move_to_start_vel_command(self, joint_pose):
        
        _joint_pose = joint_pose
        '''
          1. execute FK (joint pose -> ee pose) for start pose of robot
          2. get Cartesian pose of the robot
          3. 
        '''
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
                       block_pose=Pose(position=Point(x=0.4225, y=0.1265, z=1.1)),
                       block_reference_frame="world",
                       kinect_pose=Pose(position=Point(x=1.50, y=0.0, z=1.50)),
                       kinect_reference_frame="world"
                       ):



    # Get Models' Path
    model_path = rospkg.RosPack().get_path('sawyer_sim_examples')+"/models/"
    # Load Table SDF
    table_xml = ''
    with open (model_path + "cafe_table/model.sdf", "r") as table_file:
        table_xml=table_file.read().replace('\n', '')
    #Load kinect SDF
    # kinect_xml = ''
    # with open (model_path + "kinect/model.sdf", "r") as kinect_file:
    #     kinect_xml=kinect_file.read().replace('\n', '')
    # # Load Block URDF
    # block_xml = ''
    # with open (model_path + "block/model_1.urdf", "r") as block_file:
    #     block_xml=block_file.read().replace('\n', '')
    # # Spawn Table SDF
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp_sdf1 = spawn_sdf("cafe_table", table_xml, "/",
                             table_pose, table_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn SDF service call failed: {0}".format(e))





 #   except rospy.ServiceException, e:
 #       rospy.logerr("Spawn SDF service call failed: {0}".format(e))
    # Spawn Block URDF
    rospy.wait_for_service('/gazebo/spawn_urdf_model')
    try:
        spawn_urdf1 = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        # resp_urdf = spawn_urdf1("block", block_xml, "/",
        #                        block_pose, block_reference_frame)
    except rospy.ServiceException, e:
        rospy.logerr("Spawn URDF service call failed: {0}".format(e))

def delete_gazebo_models():
    # This will be called on ROS Exit, deleting Gazebo models
    # Do not wait for the Gazebo Delete Model service, since
    # Gazebo should already be running. If the service is not
    # available since Gazebo has been killed, it is fine to error out
    try:
        delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp_delete = delete_model("cafe_table")
        # resp_delete = delete_model("block")
    except rospy.ServiceException, e:
        print("Delete Model service call failed: {0}".format(e))


## Domain Randomization!!
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
        spawn_urdf2 = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        resp_urdf = spawn_urdf2("block", block_xml, "/",
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



    rand_x = np.random.uniform(1.03    ,1.05)
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
    print '============================================================'
    rospy.init_node("sawyer_ddpg")
    # pub2 = rospy.Publisher('/ddpg/reset2/', Float64, queue_size=1)
    pub3 = rospy.Publisher('/ddpg/vel_start/', Float64, queue_size=1)
    pub4 = rospy.Publisher('/ddpg/vel_end/', Float64, queue_size=1)
    pub5 = rospy.Publisher('/ddpg/reset/', Float64, queue_size=1)

    # goal_obs_pub = rospy.Publisher('/ddpg/goal_obs/', GoalObs, queue_size=1)
    goal_obs_pub = rospy.Publisher('/ddpg/goal_obs/', Pose, queue_size=1)


    home_pose_dict = { 
                      'right_j0':-0.020025390625 , 
                      'right_j1':0.8227529296875,
                      'right_j2':-2.0955126953125, 
                      'right_j3':2.172509765625,
                      'right_j4':0.7021171875,
                      'right_j5' :-1.5003603515625,
                      'right_j6' : -2.204990234375}



    starting_joint_angles = {'right_j0': -0.041662954890248294,
                             'right_j1': -1.0258291091425074,
                             'right_j2': 0.0293680414401436,
                             'right_j3': 1.37518162913313,
                             'right_j4':  -0.06703022873354225,
                             'right_j5': 0.7968371433926965,
                             'right_j6': 1.7659649178699421}
    limb = 'right'
    hover_distance = 0.15 # meters
    # Load Gazebo Models via Spawning Services
    # Note that the models reference is the /world frame
    # and the IK operates with respect to the /base frame
    pnp = PickAndPlace(limb, hover_distance)
    # pnp.move_to_start(starting_joint_angles)
    # load_gazebo_models()
    # Remove models from the scene on shutdown
    rospy.on_shutdown(delete_gazebo_models)

    # Starting Joint angles for right arm

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
    idx = 0
    rate = rospy.Rate(1)
    #0.7825
    epi0 = True
    # load_kinect_camera()
    # load_gazebo_models()
    # load_gazebo_block(block_poses[0])
    # delete_kinect_camera()
    # delete_gazebo_models() # what's wrong?
    # delete_gazebo_block()
    delete_gazebo_block()

    while not rospy.is_shutdown():
        # if count == 0:
        #     rospy.wait_for_message("/ddpg/epi0", JointCommand)
        # else:
        '''
            'right_j0': -0.041662954890248294,
             'right_j1': -1.0258291091425074,
             'right_j2': 0.0293680414401436,
             'right_j3': 1.67518162913313,
             'right_j4':  -0.06703022873354225,
             'right_j5': 0.7968371433926965,
             'right_j6': 1.7659649178699421

        '''
        # pnp.move_to_start(home_pose_dict)
        starting_joint_angles['right_j0'] = np.random.uniform(-0.05, 0.05)
        starting_joint_angles['right_j1'] = np.random.uniform(-0.95, -0.85)
        starting_joint_angles['right_j2'] = np.random.uniform(-0.1, 0.1)
        starting_joint_angles['right_j3'] = np.random.uniform(1.6, 1.7)
        start_pose = [starting_joint_angles['right_j0'], starting_joint_angles['right_j1'],
        starting_joint_angles['right_j2'], starting_joint_angles['right_j3'],
        starting_joint_angles['right_j4'], starting_joint_angles['right_j5'],
        starting_joint_angles['right_j6']]
        # if epi0: # for 0th episode, don't wait for reset command
        #     epi0=False
        #     print 'epi0'
        # else:
        print 'Waiting for reset......'
        while not rospy.is_shutdown():
            if rospy.has_param('ddpg_reset'):
                break
        rospy.delete_param('ddpg_reset')

        print 'Step1: move to home pose'
        pnp.move_to_start(starting_joint_angles)

        print 'Step2: remove all the environments'
        if not epi0:
            delete_kinect_camera()
            # delete_gazebo_models() # what's wrong?
            delete_gazebo_block()
        if epi0:
            load_gazebo_models() # why would I delete this?? LOL
            epi0 = False # from epi1, delete& reload the models 

        print 'Step3: reload learning environment'
        # rospy.sleep(0.1)
        load_kinect_camera()
        rand_x = np.random.uniform(0.45,0.63)

        rand_y = np.random.uniform(-0.20, 0.33)
        pose_block = Pose(position=Point(x=rand_x, y=rand_y, z=1.04),
                    orientation=overhead_orientation)

        # pnp.move_to_start(starting_joint_angles)
        # ok_togo = pnp.move_to_start_vel_command(starting_joint_angles.values())
        # if ok_togo:
        rospy.sleep(1.0)


        load_gazebo_block(block_pose=pose_block)


        pose_rob = Pose(position=Point(x=rand_x, y=rand_y, z=0.08),
            orientation=overhead_orientation)
        # pnp.gripper_open()

        # publish goal observation before resetting
        # goal_obs_pub.publish(goal)
        print 'Step4: observe goal for training HER'
        goal_obs_pub.publish(pose_rob)
        # a = Float64()
        # pub2.publish(a)

        # rospy.set_param('goal_pose', "[]")
        print '===Waiting for goal_obs acquisition response==='

        while not rospy.is_shutdown():
            if pnp.isReset:
                pnp.isReset = False
                break


        print 'Step5: move to randomly generated pose'
        rospy.set_param('vel_calc', 'true')
        ok_togo = pnp.move_to_start_vel_command(start_pose)
        if rospy.has_param('vel_calc'):
            rospy.delete_param('vel_calc')

        start = Float64()
        pub5.publish(start)
        print "Start new episodic learning"


        # wait until goal observation is acquired
        # wait until goal observation is acquired

        '''
          TODO: implement the wait thread for resetting2
        '''


        pnp._limb.set_command_timeout(15)

    return 0

if __name__ == '__main__':
    sys.exit(main())