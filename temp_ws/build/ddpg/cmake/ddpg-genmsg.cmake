# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "ddpg: 1 messages, 0 services")

set(MSG_I_FLAGS "-Iddpg:/home/irobot/catkin_ws/src/ddpg/msg;-Istd_msgs:/opt/ros/kinetic/share/std_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/kinetic/share/sensor_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/kinetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(ddpg_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/irobot/catkin_ws/src/ddpg/msg/GoalObs.msg" NAME_WE)
add_custom_target(_ddpg_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "ddpg" "/home/irobot/catkin_ws/src/ddpg/msg/GoalObs.msg" "std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(ddpg
  "/home/irobot/catkin_ws/src/ddpg/msg/GoalObs.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ddpg
)

### Generating Services

### Generating Module File
_generate_module_cpp(ddpg
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ddpg
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(ddpg_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(ddpg_generate_messages ddpg_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/ddpg/msg/GoalObs.msg" NAME_WE)
add_dependencies(ddpg_generate_messages_cpp _ddpg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ddpg_gencpp)
add_dependencies(ddpg_gencpp ddpg_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ddpg_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(ddpg
  "/home/irobot/catkin_ws/src/ddpg/msg/GoalObs.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ddpg
)

### Generating Services

### Generating Module File
_generate_module_eus(ddpg
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ddpg
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(ddpg_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(ddpg_generate_messages ddpg_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/ddpg/msg/GoalObs.msg" NAME_WE)
add_dependencies(ddpg_generate_messages_eus _ddpg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ddpg_geneus)
add_dependencies(ddpg_geneus ddpg_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ddpg_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(ddpg
  "/home/irobot/catkin_ws/src/ddpg/msg/GoalObs.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ddpg
)

### Generating Services

### Generating Module File
_generate_module_lisp(ddpg
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ddpg
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(ddpg_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(ddpg_generate_messages ddpg_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/ddpg/msg/GoalObs.msg" NAME_WE)
add_dependencies(ddpg_generate_messages_lisp _ddpg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ddpg_genlisp)
add_dependencies(ddpg_genlisp ddpg_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ddpg_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(ddpg
  "/home/irobot/catkin_ws/src/ddpg/msg/GoalObs.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ddpg
)

### Generating Services

### Generating Module File
_generate_module_nodejs(ddpg
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ddpg
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(ddpg_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(ddpg_generate_messages ddpg_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/ddpg/msg/GoalObs.msg" NAME_WE)
add_dependencies(ddpg_generate_messages_nodejs _ddpg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ddpg_gennodejs)
add_dependencies(ddpg_gennodejs ddpg_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ddpg_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(ddpg
  "/home/irobot/catkin_ws/src/ddpg/msg/GoalObs.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ddpg
)

### Generating Services

### Generating Module File
_generate_module_py(ddpg
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ddpg
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(ddpg_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(ddpg_generate_messages ddpg_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/ddpg/msg/GoalObs.msg" NAME_WE)
add_dependencies(ddpg_generate_messages_py _ddpg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(ddpg_genpy)
add_dependencies(ddpg_genpy ddpg_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS ddpg_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ddpg)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/ddpg
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(ddpg_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(ddpg_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ddpg)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/ddpg
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(ddpg_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(ddpg_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ddpg)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/ddpg
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(ddpg_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(ddpg_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ddpg)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/ddpg
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(ddpg_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(ddpg_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ddpg)
  install(CODE "execute_process(COMMAND \"/usr/bin/python\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ddpg\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/ddpg
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(ddpg_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(ddpg_generate_messages_py sensor_msgs_generate_messages_py)
endif()
