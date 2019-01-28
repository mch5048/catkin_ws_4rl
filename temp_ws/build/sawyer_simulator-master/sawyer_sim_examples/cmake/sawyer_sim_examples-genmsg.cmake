# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "sawyer_sim_examples: 2 messages, 0 services")

set(MSG_I_FLAGS "-Isawyer_sim_examples:/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg;-Istd_msgs:/opt/ros/kinetic/share/std_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/kinetic/share/sensor_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/kinetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(sawyer_sim_examples_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/PosCmd.msg" NAME_WE)
add_custom_target(_sawyer_sim_examples_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "sawyer_sim_examples" "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/PosCmd.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/GoalObs.msg" NAME_WE)
add_custom_target(_sawyer_sim_examples_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "sawyer_sim_examples" "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/GoalObs.msg" "std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(sawyer_sim_examples
  "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/PosCmd.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sawyer_sim_examples
)
_generate_msg_cpp(sawyer_sim_examples
  "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/GoalObs.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sawyer_sim_examples
)

### Generating Services

### Generating Module File
_generate_module_cpp(sawyer_sim_examples
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sawyer_sim_examples
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(sawyer_sim_examples_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(sawyer_sim_examples_generate_messages sawyer_sim_examples_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/PosCmd.msg" NAME_WE)
add_dependencies(sawyer_sim_examples_generate_messages_cpp _sawyer_sim_examples_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/GoalObs.msg" NAME_WE)
add_dependencies(sawyer_sim_examples_generate_messages_cpp _sawyer_sim_examples_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sawyer_sim_examples_gencpp)
add_dependencies(sawyer_sim_examples_gencpp sawyer_sim_examples_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sawyer_sim_examples_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(sawyer_sim_examples
  "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/PosCmd.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sawyer_sim_examples
)
_generate_msg_eus(sawyer_sim_examples
  "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/GoalObs.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sawyer_sim_examples
)

### Generating Services

### Generating Module File
_generate_module_eus(sawyer_sim_examples
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sawyer_sim_examples
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(sawyer_sim_examples_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(sawyer_sim_examples_generate_messages sawyer_sim_examples_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/PosCmd.msg" NAME_WE)
add_dependencies(sawyer_sim_examples_generate_messages_eus _sawyer_sim_examples_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/GoalObs.msg" NAME_WE)
add_dependencies(sawyer_sim_examples_generate_messages_eus _sawyer_sim_examples_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sawyer_sim_examples_geneus)
add_dependencies(sawyer_sim_examples_geneus sawyer_sim_examples_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sawyer_sim_examples_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(sawyer_sim_examples
  "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/PosCmd.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sawyer_sim_examples
)
_generate_msg_lisp(sawyer_sim_examples
  "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/GoalObs.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sawyer_sim_examples
)

### Generating Services

### Generating Module File
_generate_module_lisp(sawyer_sim_examples
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sawyer_sim_examples
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(sawyer_sim_examples_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(sawyer_sim_examples_generate_messages sawyer_sim_examples_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/PosCmd.msg" NAME_WE)
add_dependencies(sawyer_sim_examples_generate_messages_lisp _sawyer_sim_examples_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/GoalObs.msg" NAME_WE)
add_dependencies(sawyer_sim_examples_generate_messages_lisp _sawyer_sim_examples_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sawyer_sim_examples_genlisp)
add_dependencies(sawyer_sim_examples_genlisp sawyer_sim_examples_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sawyer_sim_examples_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(sawyer_sim_examples
  "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/PosCmd.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sawyer_sim_examples
)
_generate_msg_nodejs(sawyer_sim_examples
  "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/GoalObs.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sawyer_sim_examples
)

### Generating Services

### Generating Module File
_generate_module_nodejs(sawyer_sim_examples
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sawyer_sim_examples
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(sawyer_sim_examples_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(sawyer_sim_examples_generate_messages sawyer_sim_examples_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/PosCmd.msg" NAME_WE)
add_dependencies(sawyer_sim_examples_generate_messages_nodejs _sawyer_sim_examples_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/GoalObs.msg" NAME_WE)
add_dependencies(sawyer_sim_examples_generate_messages_nodejs _sawyer_sim_examples_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sawyer_sim_examples_gennodejs)
add_dependencies(sawyer_sim_examples_gennodejs sawyer_sim_examples_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sawyer_sim_examples_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(sawyer_sim_examples
  "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/PosCmd.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sawyer_sim_examples
)
_generate_msg_py(sawyer_sim_examples
  "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/GoalObs.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sawyer_sim_examples
)

### Generating Services

### Generating Module File
_generate_module_py(sawyer_sim_examples
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sawyer_sim_examples
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(sawyer_sim_examples_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(sawyer_sim_examples_generate_messages sawyer_sim_examples_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/PosCmd.msg" NAME_WE)
add_dependencies(sawyer_sim_examples_generate_messages_py _sawyer_sim_examples_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg/GoalObs.msg" NAME_WE)
add_dependencies(sawyer_sim_examples_generate_messages_py _sawyer_sim_examples_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(sawyer_sim_examples_genpy)
add_dependencies(sawyer_sim_examples_genpy sawyer_sim_examples_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS sawyer_sim_examples_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sawyer_sim_examples)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/sawyer_sim_examples
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(sawyer_sim_examples_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(sawyer_sim_examples_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sawyer_sim_examples)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/sawyer_sim_examples
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(sawyer_sim_examples_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(sawyer_sim_examples_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sawyer_sim_examples)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/sawyer_sim_examples
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(sawyer_sim_examples_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(sawyer_sim_examples_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sawyer_sim_examples)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/sawyer_sim_examples
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(sawyer_sim_examples_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(sawyer_sim_examples_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sawyer_sim_examples)
  install(CODE "execute_process(COMMAND \"/usr/bin/python\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sawyer_sim_examples\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/sawyer_sim_examples
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(sawyer_sim_examples_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(sawyer_sim_examples_generate_messages_py sensor_msgs_generate_messages_py)
endif()
