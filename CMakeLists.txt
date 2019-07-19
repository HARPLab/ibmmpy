
cmake_minimum_required(VERSION 2.8.3)
project(ibmmpy)

# First, set up the ROS Catkin package settings.  
# This will find a virtual package called 'catkin' that contains includes 
# and libraries aggregated from all the ROS catkin packages you depend on.
# 
# This macro creates:
# catkin_INCLUDE_DIRS, catkin_LIBRARIES, catkin_LIBRARY_DIRS

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  message_generation
  actionlib
  actionlib_msgs
            )
 
# Add ALL the includes we need to build: stuff from catkin AND system dependencies.
include_directories(
    include/
    ${catkin_INCLUDE_DIRS}
    ${openrave_INCLUDE_DIRS}
)

catkin_python_setup()

add_message_files(
  FILES
  GazeDataPoint.msg
  GazeData.msg
  FixationDataPoint.msg
  FixationData.msg
)

add_action_files(
    FILES
    Detector.action
)

generate_messages(
    DEPENDENCIES
    std_msgs
    geometry_msgs
    actionlib_msgs
)
# Set up the ROS Catkin package settings.
catkin_package()
 
