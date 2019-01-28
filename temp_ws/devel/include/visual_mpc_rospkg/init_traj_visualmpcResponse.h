// Generated by gencpp from file visual_mpc_rospkg/init_traj_visualmpcResponse.msg
// DO NOT EDIT!


#ifndef VISUAL_MPC_ROSPKG_MESSAGE_INIT_TRAJ_VISUALMPCRESPONSE_H
#define VISUAL_MPC_ROSPKG_MESSAGE_INIT_TRAJ_VISUALMPCRESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace visual_mpc_rospkg
{
template <class ContainerAllocator>
struct init_traj_visualmpcResponse_
{
  typedef init_traj_visualmpcResponse_<ContainerAllocator> Type;

  init_traj_visualmpcResponse_()
    {
    }
  init_traj_visualmpcResponse_(const ContainerAllocator& _alloc)
    {
  (void)_alloc;
    }







  typedef boost::shared_ptr< ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> const> ConstPtr;

}; // struct init_traj_visualmpcResponse_

typedef ::visual_mpc_rospkg::init_traj_visualmpcResponse_<std::allocator<void> > init_traj_visualmpcResponse;

typedef boost::shared_ptr< ::visual_mpc_rospkg::init_traj_visualmpcResponse > init_traj_visualmpcResponsePtr;
typedef boost::shared_ptr< ::visual_mpc_rospkg::init_traj_visualmpcResponse const> init_traj_visualmpcResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace visual_mpc_rospkg

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d41d8cd98f00b204e9800998ecf8427e";
  }

  static const char* value(const ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd41d8cd98f00b204ULL;
  static const uint64_t static_value2 = 0xe9800998ecf8427eULL;
};

template<class ContainerAllocator>
struct DataType< ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "visual_mpc_rospkg/init_traj_visualmpcResponse";
  }

  static const char* value(const ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n\
";
  }

  static const char* value(const ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream&, T)
    {}

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct init_traj_visualmpcResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream&, const std::string&, const ::visual_mpc_rospkg::init_traj_visualmpcResponse_<ContainerAllocator>&)
  {}
};

} // namespace message_operations
} // namespace ros

#endif // VISUAL_MPC_ROSPKG_MESSAGE_INIT_TRAJ_VISUALMPCRESPONSE_H
