cmake_minimum_required(VERSION 2.8.3)
project(odom_cam_track)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  image_transport
  roscpp
  sensor_msgs
  geometry_msgs
  tf
  cv_bridge
  message_generation
  std_msgs
  cmake_modules
  nodelet
)

find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)
add_message_files(
  FILES
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS
  LIBRARIES
  CATKIN_DEPENDS image_transport roscpp sensor_msgs geometry_msgs tf cv_bridge message_runtime std_msgs nodelet
  DEPENDS Eigen
)

###########
## Build ##
###########
include_directories(include  ${catkin_INCLUDE_DIRS})
include_directories(
  ${Eigen_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)



#############
## Install ##
#############


## Mark cpp header files for installation
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(FILES nodelet_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
