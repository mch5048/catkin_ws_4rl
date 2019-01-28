# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "visual_mpc_rospkg: 0 messages, 2 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/kinetic/share/std_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/kinetic/share/sensor_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/kinetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(visual_mpc_rospkg_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/init_traj_visualmpc.srv" NAME_WE)
add_custom_target(_visual_mpc_rospkg_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "visual_mpc_rospkg" "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/init_traj_visualmpc.srv" "sensor_msgs/Image:std_msgs/Header"
)

get_filename_component(_filename "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/get_action.srv" NAME_WE)
add_custom_target(_visual_mpc_rospkg_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "visual_mpc_rospkg" "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/get_action.srv" "sensor_msgs/Image:std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(visual_mpc_rospkg
  "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/init_traj_visualmpc.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/visual_mpc_rospkg
)
_generate_srv_cpp(visual_mpc_rospkg
  "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/get_action.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/visual_mpc_rospkg
)

### Generating Module File
_generate_module_cpp(visual_mpc_rospkg
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/visual_mpc_rospkg
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(visual_mpc_rospkg_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(visual_mpc_rospkg_generate_messages visual_mpc_rospkg_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/init_traj_visualmpc.srv" NAME_WE)
add_dependencies(visual_mpc_rospkg_generate_messages_cpp _visual_mpc_rospkg_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/get_action.srv" NAME_WE)
add_dependencies(visual_mpc_rospkg_generate_messages_cpp _visual_mpc_rospkg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(visual_mpc_rospkg_gencpp)
add_dependencies(visual_mpc_rospkg_gencpp visual_mpc_rospkg_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS visual_mpc_rospkg_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(visual_mpc_rospkg
  "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/init_traj_visualmpc.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/visual_mpc_rospkg
)
_generate_srv_eus(visual_mpc_rospkg
  "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/get_action.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/visual_mpc_rospkg
)

### Generating Module File
_generate_module_eus(visual_mpc_rospkg
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/visual_mpc_rospkg
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(visual_mpc_rospkg_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(visual_mpc_rospkg_generate_messages visual_mpc_rospkg_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/init_traj_visualmpc.srv" NAME_WE)
add_dependencies(visual_mpc_rospkg_generate_messages_eus _visual_mpc_rospkg_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/get_action.srv" NAME_WE)
add_dependencies(visual_mpc_rospkg_generate_messages_eus _visual_mpc_rospkg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(visual_mpc_rospkg_geneus)
add_dependencies(visual_mpc_rospkg_geneus visual_mpc_rospkg_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS visual_mpc_rospkg_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(visual_mpc_rospkg
  "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/init_traj_visualmpc.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/visual_mpc_rospkg
)
_generate_srv_lisp(visual_mpc_rospkg
  "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/get_action.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/visual_mpc_rospkg
)

### Generating Module File
_generate_module_lisp(visual_mpc_rospkg
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/visual_mpc_rospkg
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(visual_mpc_rospkg_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(visual_mpc_rospkg_generate_messages visual_mpc_rospkg_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/init_traj_visualmpc.srv" NAME_WE)
add_dependencies(visual_mpc_rospkg_generate_messages_lisp _visual_mpc_rospkg_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/get_action.srv" NAME_WE)
add_dependencies(visual_mpc_rospkg_generate_messages_lisp _visual_mpc_rospkg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(visual_mpc_rospkg_genlisp)
add_dependencies(visual_mpc_rospkg_genlisp visual_mpc_rospkg_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS visual_mpc_rospkg_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(visual_mpc_rospkg
  "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/init_traj_visualmpc.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/visual_mpc_rospkg
)
_generate_srv_nodejs(visual_mpc_rospkg
  "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/get_action.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/visual_mpc_rospkg
)

### Generating Module File
_generate_module_nodejs(visual_mpc_rospkg
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/visual_mpc_rospkg
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(visual_mpc_rospkg_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(visual_mpc_rospkg_generate_messages visual_mpc_rospkg_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/init_traj_visualmpc.srv" NAME_WE)
add_dependencies(visual_mpc_rospkg_generate_messages_nodejs _visual_mpc_rospkg_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/get_action.srv" NAME_WE)
add_dependencies(visual_mpc_rospkg_generate_messages_nodejs _visual_mpc_rospkg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(visual_mpc_rospkg_gennodejs)
add_dependencies(visual_mpc_rospkg_gennodejs visual_mpc_rospkg_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS visual_mpc_rospkg_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(visual_mpc_rospkg
  "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/init_traj_visualmpc.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/visual_mpc_rospkg
)
_generate_srv_py(visual_mpc_rospkg
  "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/get_action.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/visual_mpc_rospkg
)

### Generating Module File
_generate_module_py(visual_mpc_rospkg
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/visual_mpc_rospkg
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(visual_mpc_rospkg_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(visual_mpc_rospkg_generate_messages visual_mpc_rospkg_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/init_traj_visualmpc.srv" NAME_WE)
add_dependencies(visual_mpc_rospkg_generate_messages_py _visual_mpc_rospkg_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/irobot/catkin_ws/src/visual_mpc/python_visual_mpc/sawyer/visual_mpc_rospkg/srv/get_action.srv" NAME_WE)
add_dependencies(visual_mpc_rospkg_generate_messages_py _visual_mpc_rospkg_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(visual_mpc_rospkg_genpy)
add_dependencies(visual_mpc_rospkg_genpy visual_mpc_rospkg_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS visual_mpc_rospkg_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/visual_mpc_rospkg)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/visual_mpc_rospkg
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(visual_mpc_rospkg_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(visual_mpc_rospkg_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/visual_mpc_rospkg)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/visual_mpc_rospkg
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(visual_mpc_rospkg_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(visual_mpc_rospkg_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/visual_mpc_rospkg)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/visual_mpc_rospkg
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(visual_mpc_rospkg_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(visual_mpc_rospkg_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/visual_mpc_rospkg)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/visual_mpc_rospkg
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(visual_mpc_rospkg_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(visual_mpc_rospkg_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/visual_mpc_rospkg)
  install(CODE "execute_process(COMMAND \"/usr/bin/python\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/visual_mpc_rospkg\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/visual_mpc_rospkg
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(visual_mpc_rospkg_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(visual_mpc_rospkg_generate_messages_py sensor_msgs_generate_messages_py)
endif()
