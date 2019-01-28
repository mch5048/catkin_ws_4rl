# CMake generated Testfile for 
# Source directory: /home/irobot/catkin_ws/src/urdf-kinetic-devel/urdf
# Build directory: /home/irobot/catkin_ws/build/urdf-kinetic-devel/urdf
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(_ctest_urdf_rostest_test_test_robot_model_parser.launch "/home/irobot/catkin_ws/build/catkin_generated/env_cached.sh" "/usr/bin/python" "/opt/ros/kinetic/share/catkin/cmake/test/run_tests.py" "/home/irobot/catkin_ws/build/test_results/urdf/rostest-test_test_robot_model_parser.xml" "--return-code" "/opt/ros/kinetic/share/rostest/cmake/../../../bin/rostest --pkgdir=/home/irobot/catkin_ws/src/urdf-kinetic-devel/urdf --package=urdf --results-filename test_test_robot_model_parser.xml --results-base-dir \"/home/irobot/catkin_ws/build/test_results\" /home/irobot/catkin_ws/src/urdf-kinetic-devel/urdf/test/test_robot_model_parser.launch ")
add_test(_ctest_urdf_gtest_urdfdom_compatibility_test "/home/irobot/catkin_ws/build/catkin_generated/env_cached.sh" "/usr/bin/python" "/opt/ros/kinetic/share/catkin/cmake/test/run_tests.py" "/home/irobot/catkin_ws/build/test_results/urdf/gtest-urdfdom_compatibility_test.xml" "--return-code" "/home/irobot/catkin_ws/devel/lib/urdf/urdfdom_compatibility_test --gtest_output=xml:/home/irobot/catkin_ws/build/test_results/urdf/gtest-urdfdom_compatibility_test.xml")
