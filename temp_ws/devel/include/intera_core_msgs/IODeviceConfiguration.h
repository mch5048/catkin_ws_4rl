// Generated by gencpp from file intera_core_msgs/IODeviceConfiguration.msg
// DO NOT EDIT!


#ifndef INTERA_CORE_MSGS_MESSAGE_IODEVICECONFIGURATION_H
#define INTERA_CORE_MSGS_MESSAGE_IODEVICECONFIGURATION_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <intera_core_msgs/IOComponentConfiguration.h>
#include <intera_core_msgs/IOComponentConfiguration.h>
#include <intera_core_msgs/IOComponentConfiguration.h>

namespace intera_core_msgs
{
template <class ContainerAllocator>
struct IODeviceConfiguration_
{
  typedef IODeviceConfiguration_<ContainerAllocator> Type;

  IODeviceConfiguration_()
    : time()
    , commanded()
    , upgraded()
    , device()
    , ports()
    , signals()  {
    }
  IODeviceConfiguration_(const ContainerAllocator& _alloc)
    : time()
    , commanded(_alloc)
    , upgraded(_alloc)
    , device(_alloc)
    , ports(_alloc)
    , signals(_alloc)  {
  (void)_alloc;
    }



   typedef ros::Time _time_type;
  _time_type time;

   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _commanded_type;
  _commanded_type commanded;

   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _upgraded_type;
  _upgraded_type upgraded;

   typedef  ::intera_core_msgs::IOComponentConfiguration_<ContainerAllocator>  _device_type;
  _device_type device;

   typedef std::vector< ::intera_core_msgs::IOComponentConfiguration_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::intera_core_msgs::IOComponentConfiguration_<ContainerAllocator> >::other >  _ports_type;
  _ports_type ports;

   typedef std::vector< ::intera_core_msgs::IOComponentConfiguration_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::intera_core_msgs::IOComponentConfiguration_<ContainerAllocator> >::other >  _signals_type;
  _signals_type signals;





  typedef boost::shared_ptr< ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> const> ConstPtr;

}; // struct IODeviceConfiguration_

typedef ::intera_core_msgs::IODeviceConfiguration_<std::allocator<void> > IODeviceConfiguration;

typedef boost::shared_ptr< ::intera_core_msgs::IODeviceConfiguration > IODeviceConfigurationPtr;
typedef boost::shared_ptr< ::intera_core_msgs::IODeviceConfiguration const> IODeviceConfigurationConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> >::stream(s, "", v);
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
struct IsFixedSize< ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> >
{
  static const char* value()
  {
    return "6757fad6217033498191470cb08f1674";
  }

  static const char* value(const ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x6757fad621703349ULL;
  static const uint64_t static_value2 = 0x8191470cb08f1674ULL;
};

template<class ContainerAllocator>
struct DataType< ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> >
{
  static const char* value()
  {
    return "intera_core_msgs/IODeviceConfiguration";
  }

  static const char* value(const ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> >
{
  static const char* value()
  {
    return "## IO Device Configuration\n\
time time                             # configuration command timestamp\n\
string commanded                      # configuration command JSON\n\
string upgraded                       # configuration command JSON, upgraded to current schema revision\n\
IOComponentConfiguration   device     # instantiated device configuration\n\
IOComponentConfiguration[] ports      # instantiated port configurations\n\
IOComponentConfiguration[] signals    # instantiated signal configurations\n\
\n\
================================================================================\n\
MSG: intera_core_msgs/IOComponentConfiguration\n\
## IO Component configuration data\n\
string name                           # component name\n\
string config                         # component configuration JSON\n\
";
  }

  static const char* value(const ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.time);
      stream.next(m.commanded);
      stream.next(m.upgraded);
      stream.next(m.device);
      stream.next(m.ports);
      stream.next(m.signals);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct IODeviceConfiguration_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::intera_core_msgs::IODeviceConfiguration_<ContainerAllocator>& v)
  {
    s << indent << "time: ";
    Printer<ros::Time>::stream(s, indent + "  ", v.time);
    s << indent << "commanded: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.commanded);
    s << indent << "upgraded: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.upgraded);
    s << indent << "device: ";
    s << std::endl;
    Printer< ::intera_core_msgs::IOComponentConfiguration_<ContainerAllocator> >::stream(s, indent + "  ", v.device);
    s << indent << "ports[]" << std::endl;
    for (size_t i = 0; i < v.ports.size(); ++i)
    {
      s << indent << "  ports[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::intera_core_msgs::IOComponentConfiguration_<ContainerAllocator> >::stream(s, indent + "    ", v.ports[i]);
    }
    s << indent << "signals[]" << std::endl;
    for (size_t i = 0; i < v.signals.size(); ++i)
    {
      s << indent << "  signals[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::intera_core_msgs::IOComponentConfiguration_<ContainerAllocator> >::stream(s, indent + "    ", v.signals[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // INTERA_CORE_MSGS_MESSAGE_IODEVICECONFIGURATION_H
