// Generated by gencpp from file intera_core_msgs/SolvePositionIKRequest.msg
// DO NOT EDIT!


#ifndef INTERA_CORE_MSGS_MESSAGE_SOLVEPOSITIONIKREQUEST_H
#define INTERA_CORE_MSGS_MESSAGE_SOLVEPOSITIONIKREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/JointState.h>

namespace intera_core_msgs
{
template <class ContainerAllocator>
struct SolvePositionIKRequest_
{
  typedef SolvePositionIKRequest_<ContainerAllocator> Type;

  SolvePositionIKRequest_()
    : pose_stamp()
    , seed_angles()
    , seed_mode(0)
    , use_nullspace_goal()
    , nullspace_goal()
    , nullspace_gain()
    , tip_names()  {
    }
  SolvePositionIKRequest_(const ContainerAllocator& _alloc)
    : pose_stamp(_alloc)
    , seed_angles(_alloc)
    , seed_mode(0)
    , use_nullspace_goal(_alloc)
    , nullspace_goal(_alloc)
    , nullspace_gain(_alloc)
    , tip_names(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector< ::geometry_msgs::PoseStamped_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::geometry_msgs::PoseStamped_<ContainerAllocator> >::other >  _pose_stamp_type;
  _pose_stamp_type pose_stamp;

   typedef std::vector< ::sensor_msgs::JointState_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::sensor_msgs::JointState_<ContainerAllocator> >::other >  _seed_angles_type;
  _seed_angles_type seed_angles;

   typedef int8_t _seed_mode_type;
  _seed_mode_type seed_mode;

   typedef std::vector<uint8_t, typename ContainerAllocator::template rebind<uint8_t>::other >  _use_nullspace_goal_type;
  _use_nullspace_goal_type use_nullspace_goal;

   typedef std::vector< ::sensor_msgs::JointState_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::sensor_msgs::JointState_<ContainerAllocator> >::other >  _nullspace_goal_type;
  _nullspace_goal_type nullspace_goal;

   typedef std::vector<double, typename ContainerAllocator::template rebind<double>::other >  _nullspace_gain_type;
  _nullspace_gain_type nullspace_gain;

   typedef std::vector<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > , typename ContainerAllocator::template rebind<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::other >  _tip_names_type;
  _tip_names_type tip_names;



  enum {
    SEED_AUTO = 0,
    SEED_USER = 1,
    SEED_CURRENT = 2,
    SEED_NS_MAP = 3,
  };


  typedef boost::shared_ptr< ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> const> ConstPtr;

}; // struct SolvePositionIKRequest_

typedef ::intera_core_msgs::SolvePositionIKRequest_<std::allocator<void> > SolvePositionIKRequest;

typedef boost::shared_ptr< ::intera_core_msgs::SolvePositionIKRequest > SolvePositionIKRequestPtr;
typedef boost::shared_ptr< ::intera_core_msgs::SolvePositionIKRequest const> SolvePositionIKRequestConstPtr;

// constants requiring out of line definition

   

   

   

   



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace intera_core_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'intera_core_msgs': ['/home/irobot/catkin_ws/src/intera_common-master/intera_core_msgs/msg', '/home/irobot/catkin_ws/devel/share/intera_core_msgs/msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'actionlib_msgs': ['/opt/ros/kinetic/share/actionlib_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "b09dd99695bb18639bfea7c92d0a89ca";
  }

  static const char* value(const ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xb09dd99695bb1863ULL;
  static const uint64_t static_value2 = 0x9bfea7c92d0a89caULL;
};

template<class ContainerAllocator>
struct DataType< ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "intera_core_msgs/SolvePositionIKRequest";
  }

  static const char* value(const ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n\
geometry_msgs/PoseStamped[] pose_stamp\n\
\n\
\n\
\n\
\n\
sensor_msgs/JointState[] seed_angles\n\
\n\
\n\
\n\
\n\
\n\
int8 SEED_AUTO    = 0\n\
int8 SEED_USER    = 1\n\
int8 SEED_CURRENT = 2\n\
int8 SEED_NS_MAP  = 3\n\
\n\
int8 seed_mode\n\
\n\
\n\
bool[] use_nullspace_goal\n\
\n\
\n\
sensor_msgs/JointState[] nullspace_goal\n\
\n\
\n\
\n\
float64[] nullspace_gain\n\
\n\
\n\
string[] tip_names\n\
\n\
\n\
================================================================================\n\
MSG: geometry_msgs/PoseStamped\n\
# A Pose with reference coordinate frame and timestamp\n\
Header header\n\
Pose pose\n\
\n\
================================================================================\n\
MSG: std_msgs/Header\n\
# Standard metadata for higher-level stamped data types.\n\
# This is generally used to communicate timestamped data \n\
# in a particular coordinate frame.\n\
# \n\
# sequence ID: consecutively increasing ID \n\
uint32 seq\n\
#Two-integer timestamp that is expressed as:\n\
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n\
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n\
# time-handling sugar is provided by the client library\n\
time stamp\n\
#Frame this data is associated with\n\
# 0: no frame\n\
# 1: global frame\n\
string frame_id\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Pose\n\
# A representation of pose in free space, composed of position and orientation. \n\
Point position\n\
Quaternion orientation\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Point\n\
# This contains the position of a point in free space\n\
float64 x\n\
float64 y\n\
float64 z\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Quaternion\n\
# This represents an orientation in free space in quaternion form.\n\
\n\
float64 x\n\
float64 y\n\
float64 z\n\
float64 w\n\
\n\
================================================================================\n\
MSG: sensor_msgs/JointState\n\
# This is a message that holds data to describe the state of a set of torque controlled joints. \n\
#\n\
# The state of each joint (revolute or prismatic) is defined by:\n\
#  * the position of the joint (rad or m),\n\
#  * the velocity of the joint (rad/s or m/s) and \n\
#  * the effort that is applied in the joint (Nm or N).\n\
#\n\
# Each joint is uniquely identified by its name\n\
# The header specifies the time at which the joint states were recorded. All the joint states\n\
# in one message have to be recorded at the same time.\n\
#\n\
# This message consists of a multiple arrays, one for each part of the joint state. \n\
# The goal is to make each of the fields optional. When e.g. your joints have no\n\
# effort associated with them, you can leave the effort array empty. \n\
#\n\
# All arrays in this message should have the same size, or be empty.\n\
# This is the only way to uniquely associate the joint name with the correct\n\
# states.\n\
\n\
\n\
Header header\n\
\n\
string[] name\n\
float64[] position\n\
float64[] velocity\n\
float64[] effort\n\
";
  }

  static const char* value(const ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.pose_stamp);
      stream.next(m.seed_angles);
      stream.next(m.seed_mode);
      stream.next(m.use_nullspace_goal);
      stream.next(m.nullspace_goal);
      stream.next(m.nullspace_gain);
      stream.next(m.tip_names);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct SolvePositionIKRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::intera_core_msgs::SolvePositionIKRequest_<ContainerAllocator>& v)
  {
    s << indent << "pose_stamp[]" << std::endl;
    for (size_t i = 0; i < v.pose_stamp.size(); ++i)
    {
      s << indent << "  pose_stamp[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::geometry_msgs::PoseStamped_<ContainerAllocator> >::stream(s, indent + "    ", v.pose_stamp[i]);
    }
    s << indent << "seed_angles[]" << std::endl;
    for (size_t i = 0; i < v.seed_angles.size(); ++i)
    {
      s << indent << "  seed_angles[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::sensor_msgs::JointState_<ContainerAllocator> >::stream(s, indent + "    ", v.seed_angles[i]);
    }
    s << indent << "seed_mode: ";
    Printer<int8_t>::stream(s, indent + "  ", v.seed_mode);
    s << indent << "use_nullspace_goal[]" << std::endl;
    for (size_t i = 0; i < v.use_nullspace_goal.size(); ++i)
    {
      s << indent << "  use_nullspace_goal[" << i << "]: ";
      Printer<uint8_t>::stream(s, indent + "  ", v.use_nullspace_goal[i]);
    }
    s << indent << "nullspace_goal[]" << std::endl;
    for (size_t i = 0; i < v.nullspace_goal.size(); ++i)
    {
      s << indent << "  nullspace_goal[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::sensor_msgs::JointState_<ContainerAllocator> >::stream(s, indent + "    ", v.nullspace_goal[i]);
    }
    s << indent << "nullspace_gain[]" << std::endl;
    for (size_t i = 0; i < v.nullspace_gain.size(); ++i)
    {
      s << indent << "  nullspace_gain[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.nullspace_gain[i]);
    }
    s << indent << "tip_names[]" << std::endl;
    for (size_t i = 0; i < v.tip_names.size(); ++i)
    {
      s << indent << "  tip_names[" << i << "]: ";
      Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.tip_names[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // INTERA_CORE_MSGS_MESSAGE_SOLVEPOSITIONIKREQUEST_H
