# Install script for directory: /home/irobot/catkin_ws/src/berkeley_sawyer-master

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/berkeley_sawyer/srv" TYPE FILE FILES
    "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_kinectdata.srv"
    "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/save_kinectdata.srv"
    "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj.srv"
    "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/delete_traj.srv"
    "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_action.srv"
    "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj_visualmpc.srv"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/berkeley_sawyer/cmake" TYPE FILE FILES "/home/irobot/catkin_ws/build/berkeley_sawyer-master/catkin_generated/installspace/berkeley_sawyer-msg-paths.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/include/berkeley_sawyer")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/share/roseus/ros/berkeley_sawyer")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/share/common-lisp/ros/berkeley_sawyer")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/share/gennodejs/ros/berkeley_sawyer")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  execute_process(COMMAND "/usr/bin/python" -m compileall "/home/irobot/catkin_ws/devel/lib/python2.7/dist-packages/berkeley_sawyer")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/lib/python2.7/dist-packages/berkeley_sawyer")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/irobot/catkin_ws/build/berkeley_sawyer-master/catkin_generated/installspace/berkeley_sawyer.pc")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/berkeley_sawyer/cmake" TYPE FILE FILES "/home/irobot/catkin_ws/build/berkeley_sawyer-master/catkin_generated/installspace/berkeley_sawyer-msg-extras.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/berkeley_sawyer/cmake" TYPE FILE FILES
    "/home/irobot/catkin_ws/build/berkeley_sawyer-master/catkin_generated/installspace/berkeley_sawyerConfig.cmake"
    "/home/irobot/catkin_ws/build/berkeley_sawyer-master/catkin_generated/installspace/berkeley_sawyerConfig-version.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/berkeley_sawyer" TYPE FILE FILES "/home/irobot/catkin_ws/src/berkeley_sawyer-master/package.xml")
endif()

