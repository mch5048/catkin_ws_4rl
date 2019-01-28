# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "berkeley_sawyer: 0 messages, 6 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/kinetic/share/std_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/kinetic/share/sensor_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/kinetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(berkeley_sawyer_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/delete_traj.srv" NAME_WE)
add_custom_target(_berkeley_sawyer_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "berkeley_sawyer" "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/delete_traj.srv" ""
)

get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/save_kinectdata.srv" NAME_WE)
add_custom_target(_berkeley_sawyer_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "berkeley_sawyer" "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/save_kinectdata.srv" ""
)

get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_kinectdata.srv" NAME_WE)
add_custom_target(_berkeley_sawyer_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "berkeley_sawyer" "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_kinectdata.srv" "sensor_msgs/Image:std_msgs/Header"
)

get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj.srv" NAME_WE)
add_custom_target(_berkeley_sawyer_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "berkeley_sawyer" "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj.srv" ""
)

get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_action.srv" NAME_WE)
add_custom_target(_berkeley_sawyer_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "berkeley_sawyer" "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_action.srv" "sensor_msgs/Image:std_msgs/Header"
)

get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj_visualmpc.srv" NAME_WE)
add_custom_target(_berkeley_sawyer_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "berkeley_sawyer" "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj_visualmpc.srv" "sensor_msgs/Image:std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/delete_traj.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_cpp(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/save_kinectdata.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_cpp(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_kinectdata.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_cpp(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_cpp(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_action.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_cpp(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj_visualmpc.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/berkeley_sawyer
)

### Generating Module File
_generate_module_cpp(berkeley_sawyer
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/berkeley_sawyer
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(berkeley_sawyer_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(berkeley_sawyer_generate_messages berkeley_sawyer_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/delete_traj.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_cpp _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/save_kinectdata.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_cpp _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_kinectdata.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_cpp _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_cpp _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_action.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_cpp _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj_visualmpc.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_cpp _berkeley_sawyer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(berkeley_sawyer_gencpp)
add_dependencies(berkeley_sawyer_gencpp berkeley_sawyer_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS berkeley_sawyer_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/delete_traj.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_eus(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/save_kinectdata.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_eus(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_kinectdata.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_eus(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_eus(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_action.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_eus(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj_visualmpc.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/berkeley_sawyer
)

### Generating Module File
_generate_module_eus(berkeley_sawyer
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/berkeley_sawyer
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(berkeley_sawyer_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(berkeley_sawyer_generate_messages berkeley_sawyer_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/delete_traj.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_eus _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/save_kinectdata.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_eus _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_kinectdata.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_eus _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_eus _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_action.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_eus _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj_visualmpc.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_eus _berkeley_sawyer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(berkeley_sawyer_geneus)
add_dependencies(berkeley_sawyer_geneus berkeley_sawyer_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS berkeley_sawyer_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/delete_traj.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_lisp(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/save_kinectdata.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_lisp(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_kinectdata.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_lisp(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_lisp(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_action.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_lisp(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj_visualmpc.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/berkeley_sawyer
)

### Generating Module File
_generate_module_lisp(berkeley_sawyer
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/berkeley_sawyer
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(berkeley_sawyer_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(berkeley_sawyer_generate_messages berkeley_sawyer_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/delete_traj.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_lisp _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/save_kinectdata.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_lisp _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_kinectdata.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_lisp _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_lisp _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_action.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_lisp _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj_visualmpc.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_lisp _berkeley_sawyer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(berkeley_sawyer_genlisp)
add_dependencies(berkeley_sawyer_genlisp berkeley_sawyer_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS berkeley_sawyer_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/delete_traj.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_nodejs(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/save_kinectdata.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_nodejs(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_kinectdata.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_nodejs(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_nodejs(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_action.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_nodejs(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj_visualmpc.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/berkeley_sawyer
)

### Generating Module File
_generate_module_nodejs(berkeley_sawyer
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/berkeley_sawyer
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(berkeley_sawyer_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(berkeley_sawyer_generate_messages berkeley_sawyer_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/delete_traj.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_nodejs _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/save_kinectdata.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_nodejs _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_kinectdata.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_nodejs _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_nodejs _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_action.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_nodejs _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj_visualmpc.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_nodejs _berkeley_sawyer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(berkeley_sawyer_gennodejs)
add_dependencies(berkeley_sawyer_gennodejs berkeley_sawyer_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS berkeley_sawyer_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/delete_traj.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_py(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/save_kinectdata.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_py(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_kinectdata.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_py(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_py(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_action.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/berkeley_sawyer
)
_generate_srv_py(berkeley_sawyer
  "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj_visualmpc.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/berkeley_sawyer
)

### Generating Module File
_generate_module_py(berkeley_sawyer
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/berkeley_sawyer
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(berkeley_sawyer_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(berkeley_sawyer_generate_messages berkeley_sawyer_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/delete_traj.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_py _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/save_kinectdata.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_py _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_kinectdata.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_py _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_py _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/get_action.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_py _berkeley_sawyer_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/berkeley_sawyer-master/srv/init_traj_visualmpc.srv" NAME_WE)
add_dependencies(berkeley_sawyer_generate_messages_py _berkeley_sawyer_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(berkeley_sawyer_genpy)
add_dependencies(berkeley_sawyer_genpy berkeley_sawyer_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS berkeley_sawyer_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/berkeley_sawyer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/berkeley_sawyer
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(berkeley_sawyer_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(berkeley_sawyer_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/berkeley_sawyer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/berkeley_sawyer
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(berkeley_sawyer_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(berkeley_sawyer_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/berkeley_sawyer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/berkeley_sawyer
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(berkeley_sawyer_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(berkeley_sawyer_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/berkeley_sawyer)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/berkeley_sawyer
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(berkeley_sawyer_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(berkeley_sawyer_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/berkeley_sawyer)
  install(CODE "execute_process(COMMAND \"/usr/bin/python\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/berkeley_sawyer\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/berkeley_sawyer
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(berkeley_sawyer_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(berkeley_sawyer_generate_messages_py sensor_msgs_generate_messages_py)
endif()
