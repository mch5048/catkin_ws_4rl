# Install script for directory: /home/irobot/catkin_ws/src/control_msgs/control_msgs

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/control_msgs/action" TYPE FILE FILES
    "/home/irobot/catkin_ws/src/control_msgs/control_msgs/action/FollowJointTrajectory.action"
    "/home/irobot/catkin_ws/src/control_msgs/control_msgs/action/GripperCommand.action"
    "/home/irobot/catkin_ws/src/control_msgs/control_msgs/action/JointTrajectory.action"
    "/home/irobot/catkin_ws/src/control_msgs/control_msgs/action/PointHead.action"
    "/home/irobot/catkin_ws/src/control_msgs/control_msgs/action/SingleJointPosition.action"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/control_msgs/msg" TYPE FILE FILES
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/FollowJointTrajectoryAction.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/FollowJointTrajectoryActionGoal.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/FollowJointTrajectoryActionResult.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/FollowJointTrajectoryActionFeedback.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/FollowJointTrajectoryGoal.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/FollowJointTrajectoryResult.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/FollowJointTrajectoryFeedback.msg"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/control_msgs/msg" TYPE FILE FILES
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/GripperCommandAction.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/GripperCommandActionGoal.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/GripperCommandActionResult.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/GripperCommandActionFeedback.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/GripperCommandGoal.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/GripperCommandResult.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/GripperCommandFeedback.msg"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/control_msgs/msg" TYPE FILE FILES
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/JointTrajectoryAction.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/JointTrajectoryActionGoal.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/JointTrajectoryActionResult.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/JointTrajectoryActionFeedback.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/JointTrajectoryGoal.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/JointTrajectoryResult.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/JointTrajectoryFeedback.msg"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/control_msgs/msg" TYPE FILE FILES
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/PointHeadAction.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/PointHeadActionGoal.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/PointHeadActionResult.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/PointHeadActionFeedback.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/PointHeadGoal.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/PointHeadResult.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/PointHeadFeedback.msg"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/control_msgs/msg" TYPE FILE FILES
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/SingleJointPositionAction.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/SingleJointPositionActionGoal.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/SingleJointPositionActionResult.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/SingleJointPositionActionFeedback.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/SingleJointPositionGoal.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/SingleJointPositionResult.msg"
    "/home/irobot/catkin_ws/devel/share/control_msgs/msg/SingleJointPositionFeedback.msg"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/control_msgs/msg" TYPE FILE FILES
    "/home/irobot/catkin_ws/src/control_msgs/control_msgs/msg/GripperCommand.msg"
    "/home/irobot/catkin_ws/src/control_msgs/control_msgs/msg/JointControllerState.msg"
    "/home/irobot/catkin_ws/src/control_msgs/control_msgs/msg/JointTolerance.msg"
    "/home/irobot/catkin_ws/src/control_msgs/control_msgs/msg/JointTrajectoryControllerState.msg"
    "/home/irobot/catkin_ws/src/control_msgs/control_msgs/msg/PidState.msg"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/control_msgs/srv" TYPE FILE FILES
    "/home/irobot/catkin_ws/src/control_msgs/control_msgs/srv/QueryCalibrationState.srv"
    "/home/irobot/catkin_ws/src/control_msgs/control_msgs/srv/QueryTrajectoryState.srv"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/control_msgs/cmake" TYPE FILE FILES "/home/irobot/catkin_ws/build/control_msgs/control_msgs/catkin_generated/installspace/control_msgs-msg-paths.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/include/control_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/share/roseus/ros/control_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/share/common-lisp/ros/control_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/share/gennodejs/ros/control_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  execute_process(COMMAND "/usr/bin/python" -m compileall "/home/irobot/catkin_ws/devel/lib/python2.7/dist-packages/control_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages" TYPE DIRECTORY FILES "/home/irobot/catkin_ws/devel/lib/python2.7/dist-packages/control_msgs")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/irobot/catkin_ws/build/control_msgs/control_msgs/catkin_generated/installspace/control_msgs.pc")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/control_msgs/cmake" TYPE FILE FILES "/home/irobot/catkin_ws/build/control_msgs/control_msgs/catkin_generated/installspace/control_msgs-msg-extras.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/control_msgs/cmake" TYPE FILE FILES
    "/home/irobot/catkin_ws/build/control_msgs/control_msgs/catkin_generated/installspace/control_msgsConfig.cmake"
    "/home/irobot/catkin_ws/build/control_msgs/control_msgs/catkin_generated/installspace/control_msgsConfig-version.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/control_msgs" TYPE FILE FILES "/home/irobot/catkin_ws/src/control_msgs/control_msgs/package.xml")
endif()

