# Install script for directory: /home/irobot/catkin_ws/src/gazebo_domain_randomization-master/gazebo_ext_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/irobot/catkin_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gazebo_ext_msgs/srv" TYPE FILE FILES
    "/home/irobot/catkin_ws/src/gazebo_domain_randomization-master/gazebo_ext_msgs/srv/GetCollisionNames.srv"
    "/home/irobot/catkin_ws/src/gazebo_domain_randomization-master/gazebo_ext_msgs/srv/GetVisualNames.srv"
    "/home/irobot/catkin_ws/src/gazebo_domain_randomization-master/gazebo_ext_msgs/srv/SetSkyProperties.srv"
    "/home/irobot/catkin_ws/src/gazebo_domain_randomization-master/gazebo_ext_msgs/srv/GetSkyProperties.srv"
    "/home/irobot/catkin_ws/src/gazebo_domain_randomization-master/gazebo_ext_msgs/srv/SetLinkVisualProperties.srv"
    "/home/irobot/catkin_ws/src/gazebo_domain_randomization-master/gazebo_ext_msgs/srv/GetLinkVisualProperties.srv"
    "/home/irobot/catkin_ws/src/gazebo_domain_randomization-master/gazebo_ext_msgs/srv/SetSurfaceParams.srv"
    "/home/irobot/catkin_ws/src/gazebo_domain_randomization-master/gazebo_ext_msgs/srv/GetSurfaceParams.srv"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gazebo_ext_msgs/cmake" TYPE FILE FILES "/home/irobot/catkin_ws/build/gazebo_domain_randomization-master/gazebo_ext_msgs/catkin_generated/installspace/gazebo_ext_msgs-msg-paths.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/include/gazebo_ext_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/share/roseus/ros/gazebo_ext_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/share/common-lisp/ros/gazebo_ext_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/share/gennodejs/ros/gazebo_ext_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  execute_process(COMMAND "/usr/bin/python" -m compileall "/home/irobot/catkin_ws/devel/lib/python2.7/dist-packages/gazebo_ext_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/lib/python2.7/dist-packages/gazebo_ext_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/irobot/catkin_ws/build/gazebo_domain_randomization-master/gazebo_ext_msgs/catkin_generated/installspace/gazebo_ext_msgs.pc")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gazebo_ext_msgs/cmake" TYPE FILE FILES "/home/irobot/catkin_ws/build/gazebo_domain_randomization-master/gazebo_ext_msgs/catkin_generated/installspace/gazebo_ext_msgs-msg-extras.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gazebo_ext_msgs/cmake" TYPE FILE FILES
    "/home/irobot/catkin_ws/build/gazebo_domain_randomization-master/gazebo_ext_msgs/catkin_generated/installspace/gazebo_ext_msgsConfig.cmake"
    "/home/irobot/catkin_ws/build/gazebo_domain_randomization-master/gazebo_ext_msgs/catkin_generated/installspace/gazebo_ext_msgsConfig-version.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gazebo_ext_msgs" TYPE FILE FILES "/home/irobot/catkin_ws/src/gazebo_domain_randomization-master/gazebo_ext_msgs/package.xml")
endif()

