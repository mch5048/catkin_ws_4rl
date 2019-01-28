// Generated by gencpp from file sawyer_sim_examples/PosCmd.msg
// DO NOT EDIT!


#ifndef SAWYER_SIM_EXAMPLES_MESSAGE_POSCMD_H
#define SAWYER_SIM_EXAMPLES_MESSAGE_POSCMD_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>

namespace sawyer_sim_examples
{
template <class ContainerAllocator>
struct PosCmd_
{
  typedef PosCmd_<ContainerAllocator> Type;

  PosCmd_()
    : header()
    , goal_cart_pos()  {
    }
  PosCmd_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , goal_cart_pos(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::vector<double, typename ContainerAllocator::template rebind<double>::other >  _goal_cart_pos_type;
  _goal_cart_pos_type goal_cart_pos;





  typedef boost::shared_ptr< ::sawyer_sim_examples::PosCmd_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::sawyer_sim_examples::PosCmd_<ContainerAllocator> const> ConstPtr;

}; // struct PosCmd_

typedef ::sawyer_sim_examples::PosCmd_<std::allocator<void> > PosCmd;

typedef boost::shared_ptr< ::sawyer_sim_examples::PosCmd > PosCmdPtr;
typedef boost::shared_ptr< ::sawyer_sim_examples::PosCmd const> PosCmdConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::sawyer_sim_examples::PosCmd_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::sawyer_sim_examples::PosCmd_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace sawyer_sim_examples

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'sawyer_sim_examples': ['/home/irobot/catkin_ws/src/sawyer_simulator-master/sawyer_sim_examples/msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::sawyer_sim_examples::PosCmd_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::sawyer_sim_examples::PosCmd_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::sawyer_sim_examples::PosCmd_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::sawyer_sim_examples::PosCmd_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::sawyer_sim_examples::PosCmd_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::sawyer_sim_examples::PosCmd_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::sawyer_sim_examples::PosCmd_<ContainerAllocator> >
{
  static const char* value()
  {
    return "2a7aed1f0b63cf2c59e8c977d20aef5b";
  }

  static const char* value(const ::sawyer_sim_examples::PosCmd_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x2a7aed1f0b63cf2cULL;
  static const uint64_t static_value2 = 0x59e8c977d20aef5bULL;
};

template<class ContainerAllocator>
struct DataType< ::sawyer_sim_examples::PosCmd_<ContainerAllocator> >
{
  static const char* value()
  {
    return "sawyer_sim_examples/PosCmd";
  }

  static const char* value(const ::sawyer_sim_examples::PosCmd_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::sawyer_sim_examples::PosCmd_<ContainerAllocator> >
{
  static const char* value()
  {
    return "##the name of the output\n\
Header header\n\
\n\
float64[] goal_cart_pos\n\
\n\
##the value to set output \n\
\n\
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
";
  }

  static const char* value(const ::sawyer_sim_examples::PosCmd_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::sawyer_sim_examples::PosCmd_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.goal_cart_pos);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct PosCmd_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::sawyer_sim_examples::PosCmd_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::sawyer_sim_examples::PosCmd_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "goal_cart_pos[]" << std::endl;
    for (size_t i = 0; i < v.goal_cart_pos.size(); ++i)
    {
      s << indent << "  goal_cart_pos[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.goal_cart_pos[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // SAWYER_SIM_EXAMPLES_MESSAGE_POSCMD_H
