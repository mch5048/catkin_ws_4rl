#!/usr/bin/env python
from string import Template
import numpy as np
import rospy
from geometry_msgs.msg import Pose, Point
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest
from gazebo_msgs.srv import DeleteModel, DeleteModelRequest

_obj_sdf = """\
<?xml version='1.0'?>
<sdf version="1.4">
<model name=$model_name>
  <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>$obj_shape</geometry>
        <material>
          <script>
            <name>$material_color</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

_shape_list = {'box': Template("<box><size>$sx $sy $sz</size></box>"),
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

class ShapesRandomizer:
    def __init__(self, max_num_model=10,
                 size_range=[0.1, 0.2],
                 spawn_range=[[0.45, -0.25, 1.005], [0.65, 0.25, 1.01]]):
        rospy.init_node("shapes_randomizer")
        self._sdf_temp = Template(_obj_sdf)
        self._spawn_model = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        self._delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        self._obj_names = ['obj' + str(i) for i in range(max_num_model)]
        self._size_range = size_range
        self._spawn_range = spawn_range

    def timer_callback(self, event):
        req = SpawnModelRequest()
        obj_name = np.random.choice(self._obj_names)
        req.model_name = obj_name
        self._delete_model(DeleteModelRequest(obj_name))
        shape_no = np.random.choice(3)
        if shape_no == 0:
            sx = np.random.uniform(*self._size_range)
            sy = np.random.uniform(*self._size_range)
            sz = np.random.uniform(*self._size_range)
            shape_sdf = _shape_list['box'].substitute(sx=str(sx), sy=str(sy), sz=str(sz))
        elif shape_no == 1:
            cr = np.random.uniform(*self._size_range)
            cl = np.random.uniform(*self._size_range)
            shape_sdf = _shape_list['cylinder'].substitute(cr=str(cr), cl=str(cl))
        else:
            sr = np.random.uniform(*self._size_range)
            shape_sdf = _shape_list['sphere'].substitute(sr=str(sr))
        req.model_xml = self._sdf_temp.substitute(model_name=obj_name,
                                                  obj_shape=shape_sdf,
                                                  material_color=np.random.choice(_material_list))
        req.initial_pose = Pose(position=Point(*np.random.uniform(self._spawn_range[0],
                                                                  self._spawn_range[1])))
        rospy.loginfo("Spawn model sdf: " + req.model_xml)
        try:
            res = self._spawn_model(req)
            if not res.success:
                rospy.logwarn(res.status_message)
        except rospy.ServiceException, e:
            rospy.logerr("Service call failed: %s" % e)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        duration = float(sys.argv[1])
    else:
        duration = 1
    randomizer = ShapesRandomizer()
    rospy.Timer(rospy.Duration(duration), randomizer.timer_callback)
    rospy.spin()
