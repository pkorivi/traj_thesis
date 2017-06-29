cmake_minimum_required(VERSION 2.8.3)
project(odom_plotter)

find_package(catkin REQUIRED COMPONENTS nav_msgs std_msgs roscpp tf rospy heading_msg)

catkin_package()

find_package(message_generation)

catkin_package(CATKIN_DEPENDS message_runtime)

add_dependencies( ${catkin_EXPORTED_TARGETS})

include_directories(${catkin_INCLUDE_DIRS})


# Build the executable
#add_executable(odom_plotter src/odom_plotter.py)
# Add a build order dependency on nav_msgs
# This ensures that nav_msgs' msg headers are built before your executable
# Install the executable
install(PROGRAMS src/odom_plotter.py
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch/
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
## Mark cpp header files for installation
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
