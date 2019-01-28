#!/usr/bin/env python

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
from intera_interface import Limb
from intera_interface import CHECK_VERSION
import rospy


limb = intera_interface.Limb("right")

rospy.Subscriber('/robot/limb/right/joint_command', JointCommand , jointCommandCB)

joint_efforts = list()

def jointCommandCB(cmd):

    joint_efforts = [limb.joint_effort('right_j0'),
    limb.joint_effort('right_j1'),
    limb.joint_effort('right_j2'),
    limb.joint_effort('right_j3'),
    limb.joint_effort('right_j4'),
    limb.joint_effort('right_j5'),
    limb.joint_effort('right_j6')]
    torque_list = cmd.effort


    


def listener():
    rospy.init_node('joint_subscriber')
    rospy.Subscriber('/robot/limb/right/joint_command', JointCommand , jointCommandCB)
    rospy.spin()

if __name__ == "__main__":
    listener()