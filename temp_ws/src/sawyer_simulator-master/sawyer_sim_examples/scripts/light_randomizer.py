#!/usr/bin/env python
import numpy as np
import rospy
from std_msgs.msg import ColorRGBA
from gazebo_msgs.srv import SetLightProperties, SetLightPropertiesRequest

class LightRandomizer:
    def __init__(self, light_name='sun'):
        rospy.init_node("light_randomizer")
        self._light_name = light_name
        self._set_light = rospy.ServiceProxy('gazebo/set_light_properties', SetLightProperties)
    def timer_callback(self, event):
        req = SetLightPropertiesRequest()
        req.light_name = self._light_name
        req.diffuse = ColorRGBA(*np.random.random(4))
        req.attenuation_constant = np.random.random()
        req.attenuation_linear = np.random.random()
        req.attenuation_quadratic = np.random.random()
        rospy.loginfo("Set light parameter: " + str(req))
        try:
            res = self._set_light(req)
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
    randomizer = LightRandomizer()
    rospy.Timer(rospy.Duration(duration), randomizer.timer_callback)
    rospy.spin()
