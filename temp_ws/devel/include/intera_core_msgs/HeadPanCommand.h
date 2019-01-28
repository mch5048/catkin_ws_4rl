// Generated by gencpp from file intera_core_msgs/HeadPanCommand.msg
// DO NOT EDIT!


#ifndef INTERA_CORE_MSGS_MESSAGE_HEADPANCOMMAND_H
#define INTERA_CORE_MSGS_MESSAGE_HEADPANCOMMAND_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace intera_core_msgs
{
template <class ContainerAllocator>
struct HeadPanCommand_
{
  typedef HeadPanCommand_<ContainerAllocator> Type;

  HeadPanCommand_()
    : target(0.0)
    , speed_ratio(0.0)
    , pan_mode(0)  {
    }
  HeadPanCommand_(const ContainerAllocator& _alloc)
    : target(0.0)
    , speed_ratio(0.0)
    , pan_mode(0)  {
  (void)_alloc;
    }



   typedef float _target_type;
  _target_type target;

   typedef float _speed_ratio_type;
  _speed_ratio_type speed_ratio;

   typedef uint8_t _pan_mode_type;
  _pan_mode_type pan_mode;



  enum {
    SET_PASSIVE_MODE = 0u,
    SET_ACTIVE_MODE = 1u,
    SET_ACTIVE_CANCELLATION_MODE = 2u,
    NO_MODE_CHANGE = 3u,
  };

  static const float MAX_SPEED_RATIO;
  static const float MIN_SPEED_RATIO;

  typedef boost::shared_ptr< ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> const> ConstPtr;

}; // struct HeadPanCommand_

typedef ::intera_core_msgs::HeadPanCommand_<std::allocator<void> > HeadPanCommand;

typedef boost::shared_ptr< ::intera_core_msgs::HeadPanCommand > HeadPanCommandPtr;
typedef boost::shared_ptr< ::intera_core_msgs::HeadPanCommand const> HeadPanCommandConstPtr;

// constants requiring out of line definition

   
   template<typename ContainerAllocator> const float
      HeadPanCommand_<ContainerAllocator>::MAX_SPEED_RATIO =
        
          1.0
        
        ;
   

   
   template<typename ContainerAllocator> const float
      HeadPanCommand_<ContainerAllocator>::MIN_SPEED_RATIO =
        
          0.0
        
        ;
   

   

   

   

   



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace intera_core_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'intera_core_msgs': ['/home/irobot/catkin_ws/src/intera_common-master/intera_core_msgs/msg', '/home/irobot/catkin_ws/devel/share/intera_core_msgs/msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'actionlib_msgs': ['/opt/ros/kinetic/share/actionlib_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> >
{
  static const char* value()
  {
    return "5cb68e8755646564cf47813f91cee216";
  }

  static const char* value(const ::intera_core_msgs::HeadPanCommand_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x5cb68e8755646564ULL;
  static const uint64_t static_value2 = 0xcf47813f91cee216ULL;
};

template<class ContainerAllocator>
struct DataType< ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> >
{
  static const char* value()
  {
    return "intera_core_msgs/HeadPanCommand";
  }

  static const char* value(const ::intera_core_msgs::HeadPanCommand_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float32 target              # radians for target, 0 str\n\
float32 speed_ratio         # Percentage of max speed [0-1]\n\
#\n\
  float32 MAX_SPEED_RATIO = 1.0\n\
  float32 MIN_SPEED_RATIO = 0.0\n\
#\n\
uint8   pan_mode  # set to one of constants below to change pan mode\n\
# pan_mode possible states\n\
  uint8   SET_PASSIVE_MODE = 0\n\
  uint8   SET_ACTIVE_MODE = 1\n\
  uint8   SET_ACTIVE_CANCELLATION_MODE = 2\n\
  uint8   NO_MODE_CHANGE = 3\n\
#\n\
";
  }

  static const char* value(const ::intera_core_msgs::HeadPanCommand_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.target);
      stream.next(m.speed_ratio);
      stream.next(m.pan_mode);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct HeadPanCommand_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::intera_core_msgs::HeadPanCommand_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::intera_core_msgs::HeadPanCommand_<ContainerAllocator>& v)
  {
    s << indent << "target: ";
    Printer<float>::stream(s, indent + "  ", v.target);
    s << indent << "speed_ratio: ";
    Printer<float>::stream(s, indent + "  ", v.speed_ratio);
    s << indent << "pan_mode: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.pan_mode);
  }
};

} // namespace message_operations
} // namespace ros

#endif // INTERA_CORE_MSGS_MESSAGE_HEADPANCOMMAND_H
