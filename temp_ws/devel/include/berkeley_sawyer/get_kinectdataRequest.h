// Generated by gencpp from file berkeley_sawyer/get_kinectdataRequest.msg
// DO NOT EDIT!


#ifndef BERKELEY_SAWYER_MESSAGE_GET_KINECTDATAREQUEST_H
#define BERKELEY_SAWYER_MESSAGE_GET_KINECTDATAREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace berkeley_sawyer
{
template <class ContainerAllocator>
struct get_kinectdataRequest_
{
  typedef get_kinectdataRequest_<ContainerAllocator> Type;

  get_kinectdataRequest_()
    {
    }
  get_kinectdataRequest_(const ContainerAllocator& _alloc)
    {
  (void)_alloc;
    }







  typedef boost::shared_ptr< ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> const> ConstPtr;

}; // struct get_kinectdataRequest_

typedef ::berkeley_sawyer::get_kinectdataRequest_<std::allocator<void> > get_kinectdataRequest;

typedef boost::shared_ptr< ::berkeley_sawyer::get_kinectdataRequest > get_kinectdataRequestPtr;
typedef boost::shared_ptr< ::berkeley_sawyer::get_kinectdataRequest const> get_kinectdataRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace berkeley_sawyer

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d41d8cd98f00b204e9800998ecf8427e";
  }

  static const char* value(const ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd41d8cd98f00b204ULL;
  static const uint64_t static_value2 = 0xe9800998ecf8427eULL;
};

template<class ContainerAllocator>
struct DataType< ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "berkeley_sawyer/get_kinectdataRequest";
  }

  static const char* value(const ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n\
";
  }

  static const char* value(const ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream&, T)
    {}

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct get_kinectdataRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream&, const std::string&, const ::berkeley_sawyer::get_kinectdataRequest_<ContainerAllocator>&)
  {}
};

} // namespace message_operations
} // namespace ros

#endif // BERKELEY_SAWYER_MESSAGE_GET_KINECTDATAREQUEST_H
