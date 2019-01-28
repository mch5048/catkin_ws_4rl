execute_process(COMMAND "/home/irobot/catkin_ws/build/hrl-kdl-indigo-devel/hrl_geom/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/irobot/catkin_ws/build/hrl-kdl-indigo-devel/hrl_geom/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
