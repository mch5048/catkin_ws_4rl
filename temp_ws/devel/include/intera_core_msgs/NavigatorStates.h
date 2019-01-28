// Generated by gencpp from file intera_core_msgs/NavigatorStates.msg
// DO NOT EDIT!


#ifndef INTERA_CORE_MSGS_MESSAGE_NAVIGATORSTATES_H
#define INTERA_CORE_MSGS_MESSAGE_NAVIGATORSTATES_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <intera_core_msgs/NavigatorState.h>

namespace intera_core_msgs
{
template <class ContainerAllocator>
struct NavigatorStates_
{
  typedef NavigatorStates_<ContainerAllocator> Type;

  NavigatorStates_()
    : names()
    , states()  {
    }
  NavigatorStates_(const ContainerAllocator& _alloc)
    : names(_alloc)
    , states(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > , typename ContainerAllocator::template rebind<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::other >  _names_type;
  _names_type names;

   typedef std::vector< ::intera_core_msgs::NavigatorState_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::intera_core_msgs::NavigatorState_<ContainerAllocator> >::other >  _states_type;
  _states_type states;





  typedef boost::shared_ptr< ::intera_core_msgs::NavigatorStates_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::intera_core_msgs::NavigatorStates_<ContainerAllocator> const> ConstPtr;

}; // struct NavigatorStates_

typedef ::intera_core_msgs::NavigatorStates_<std::allocator<void> > NavigatorStates;

typedef boost::shared_ptr< ::intera_core_msgs::NavigatorStates > NavigatorStatesPtr;
typedef boost::shared_ptr< ::intera_core_msgs::NavigatorStates const> NavigatorStatesConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::intera_core_msgs::NavigatorStates_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::intera_core_msgs::NavigatorStates_<ContainerAllocator> >::stream(s, "", v);
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
struct IsFixedSize< ::intera_core_msgs::NavigatorStates_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::intera_core_msgs::NavigatorStates_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::intera_core_msgs::NavigatorStates_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::intera_core_msgs::NavigatorStates_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::intera_core_msgs::NavigatorStates_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::intera_core_msgs::NavigatorStates_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::intera_core_msgs::NavigatorStates_<ContainerAllocator> >
{
  static const char* value()
  {
    return "2c2eeb02fbbaa6f1ab6c680887f2db78";
  }

  static const char* value(const ::intera_core_msgs::NavigatorStates_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x2c2eeb02fbbaa6f1ULL;
  static const uint64_t static_value2 = 0xab6c680887f2db78ULL;
};

template<class ContainerAllocator>
struct DataType< ::intera_core_msgs::NavigatorStates_<ContainerAllocator> >
{
  static const char* value()
  {
    return "intera_core_msgs/NavigatorStates";
  }

  static const char* value(const ::intera_core_msgs::NavigatorStates_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::intera_core_msgs::NavigatorStates_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# used when publishing multiple navigators\n\
string[]         names\n\
NavigatorState[] states\n\
\n\
================================================================================\n\
MSG: intera_core_msgs/NavigatorState\n\
# buttons\n\
string[] button_names\n\
bool[] buttons\n\
\n\
# wheel position\n\
uint8   wheel\n\
\n\
# true if the light is on, false if not\n\
# lights map to button names\n\
string[] light_names\n\
bool[] lights\n\
";
  }

  static const char* value(const ::intera_core_msgs::NavigatorStates_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::intera_core_msgs::NavigatorStates_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.names);
      stream.next(m.states);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct NavigatorStates_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::intera_core_msgs::NavigatorStates_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::intera_core_msgs::NavigatorStates_<ContainerAllocator>& v)
  {
    s << indent << "names[]" << std::endl;
    for (size_t i = 0; i < v.names.size(); ++i)
    {
      s << indent << "  names[" << i << "]: ";
      Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.names[i]);
    }
    s << indent << "states[]" << std::endl;
    for (size_t i = 0; i < v.states.size(); ++i)
    {
      s << indent << "  states[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::intera_core_msgs::NavigatorState_<ContainerAllocator> >::stream(s, indent + "    ", v.states[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // INTERA_CORE_MSGS_MESSAGE_NAVIGATORSTATES_H
