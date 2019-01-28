// Generated by gencpp from file gazebo_ext_msgs/SetSkyPropertiesRequest.msg
// DO NOT EDIT!


#ifndef GAZEBO_EXT_MSGS_MESSAGE_SETSKYPROPERTIESREQUEST_H
#define GAZEBO_EXT_MSGS_MESSAGE_SETSKYPROPERTIESREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/ColorRGBA.h>

namespace gazebo_ext_msgs
{
template <class ContainerAllocator>
struct SetSkyPropertiesRequest_
{
  typedef SetSkyPropertiesRequest_<ContainerAllocator> Type;

  SetSkyPropertiesRequest_()
    : time(0.0)
    , sunrise(0.0)
    , sunset(0.0)
    , wind_speed(0.0)
    , wind_direction(0.0)
    , cloud_ambient()
    , humidity(0.0)
    , mean_cloud_size(0.0)  {
    }
  SetSkyPropertiesRequest_(const ContainerAllocator& _alloc)
    : time(0.0)
    , sunrise(0.0)
    , sunset(0.0)
    , wind_speed(0.0)
    , wind_direction(0.0)
    , cloud_ambient(_alloc)
    , humidity(0.0)
    , mean_cloud_size(0.0)  {
  (void)_alloc;
    }



   typedef double _time_type;
  _time_type time;

   typedef double _sunrise_type;
  _sunrise_type sunrise;

   typedef double _sunset_type;
  _sunset_type sunset;

   typedef double _wind_speed_type;
  _wind_speed_type wind_speed;

   typedef double _wind_direction_type;
  _wind_direction_type wind_direction;

   typedef  ::std_msgs::ColorRGBA_<ContainerAllocator>  _cloud_ambient_type;
  _cloud_ambient_type cloud_ambient;

   typedef double _humidity_type;
  _humidity_type humidity;

   typedef double _mean_cloud_size_type;
  _mean_cloud_size_type mean_cloud_size;





  typedef boost::shared_ptr< ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> const> ConstPtr;

}; // struct SetSkyPropertiesRequest_

typedef ::gazebo_ext_msgs::SetSkyPropertiesRequest_<std::allocator<void> > SetSkyPropertiesRequest;

typedef boost::shared_ptr< ::gazebo_ext_msgs::SetSkyPropertiesRequest > SetSkyPropertiesRequestPtr;
typedef boost::shared_ptr< ::gazebo_ext_msgs::SetSkyPropertiesRequest const> SetSkyPropertiesRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace gazebo_ext_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ed25a3f6c62317b873911c0baf6969fe";
  }

  static const char* value(const ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xed25a3f6c62317b8ULL;
  static const uint64_t static_value2 = 0x73911c0baf6969feULL;
};

template<class ContainerAllocator>
struct DataType< ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "gazebo_ext_msgs/SetSkyPropertiesRequest";
  }

  static const char* value(const ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float64 time\n\
float64 sunrise\n\
float64 sunset\n\
float64 wind_speed\n\
float64 wind_direction\n\
std_msgs/ColorRGBA cloud_ambient\n\
float64 humidity\n\
float64 mean_cloud_size\n\
\n\
================================================================================\n\
MSG: std_msgs/ColorRGBA\n\
float32 r\n\
float32 g\n\
float32 b\n\
float32 a\n\
";
  }

  static const char* value(const ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.time);
      stream.next(m.sunrise);
      stream.next(m.sunset);
      stream.next(m.wind_speed);
      stream.next(m.wind_direction);
      stream.next(m.cloud_ambient);
      stream.next(m.humidity);
      stream.next(m.mean_cloud_size);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct SetSkyPropertiesRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::gazebo_ext_msgs::SetSkyPropertiesRequest_<ContainerAllocator>& v)
  {
    s << indent << "time: ";
    Printer<double>::stream(s, indent + "  ", v.time);
    s << indent << "sunrise: ";
    Printer<double>::stream(s, indent + "  ", v.sunrise);
    s << indent << "sunset: ";
    Printer<double>::stream(s, indent + "  ", v.sunset);
    s << indent << "wind_speed: ";
    Printer<double>::stream(s, indent + "  ", v.wind_speed);
    s << indent << "wind_direction: ";
    Printer<double>::stream(s, indent + "  ", v.wind_direction);
    s << indent << "cloud_ambient: ";
    s << std::endl;
    Printer< ::std_msgs::ColorRGBA_<ContainerAllocator> >::stream(s, indent + "  ", v.cloud_ambient);
    s << indent << "humidity: ";
    Printer<double>::stream(s, indent + "  ", v.humidity);
    s << indent << "mean_cloud_size: ";
    Printer<double>::stream(s, indent + "  ", v.mean_cloud_size);
  }
};

} // namespace message_operations
} // namespace ros

#endif // GAZEBO_EXT_MSGS_MESSAGE_SETSKYPROPERTIESREQUEST_H
