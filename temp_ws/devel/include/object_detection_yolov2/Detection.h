// Generated by gencpp from file object_detection_yolov2/Detection.msg
// DO NOT EDIT!


#ifndef OBJECT_DETECTION_YOLOV2_MESSAGE_DETECTION_H
#define OBJECT_DETECTION_YOLOV2_MESSAGE_DETECTION_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>

namespace object_detection_yolov2
{
template <class ContainerAllocator>
struct Detection_
{
  typedef Detection_<ContainerAllocator> Type;

  Detection_()
    : header()
    , object_class()
    , p(0.0)
    , x(0)
    , y(0)
    , cam_x(0.0)
    , cam_y(0.0)
    , cam_z(0.0)
    , width(0)
    , height(0)  {
    }
  Detection_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , object_class(_alloc)
    , p(0.0)
    , x(0)
    , y(0)
    , cam_x(0.0)
    , cam_y(0.0)
    , cam_z(0.0)
    , width(0)
    , height(0)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _object_class_type;
  _object_class_type object_class;

   typedef float _p_type;
  _p_type p;

   typedef uint16_t _x_type;
  _x_type x;

   typedef uint16_t _y_type;
  _y_type y;

   typedef float _cam_x_type;
  _cam_x_type cam_x;

   typedef float _cam_y_type;
  _cam_y_type cam_y;

   typedef float _cam_z_type;
  _cam_z_type cam_z;

   typedef uint16_t _width_type;
  _width_type width;

   typedef uint16_t _height_type;
  _height_type height;





  typedef boost::shared_ptr< ::object_detection_yolov2::Detection_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::object_detection_yolov2::Detection_<ContainerAllocator> const> ConstPtr;

}; // struct Detection_

typedef ::object_detection_yolov2::Detection_<std::allocator<void> > Detection;

typedef boost::shared_ptr< ::object_detection_yolov2::Detection > DetectionPtr;
typedef boost::shared_ptr< ::object_detection_yolov2::Detection const> DetectionConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::object_detection_yolov2::Detection_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::object_detection_yolov2::Detection_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace object_detection_yolov2

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'object_detection_yolov2': ['/home/irobot/catkin_ws/src/object_detection_yolov2/msg'], 'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::object_detection_yolov2::Detection_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::object_detection_yolov2::Detection_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::object_detection_yolov2::Detection_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::object_detection_yolov2::Detection_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::object_detection_yolov2::Detection_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::object_detection_yolov2::Detection_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::object_detection_yolov2::Detection_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d82132341465f8c6318faea203b0884c";
  }

  static const char* value(const ::object_detection_yolov2::Detection_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd82132341465f8c6ULL;
  static const uint64_t static_value2 = 0x318faea203b0884cULL;
};

template<class ContainerAllocator>
struct DataType< ::object_detection_yolov2::Detection_<ContainerAllocator> >
{
  static const char* value()
  {
    return "object_detection_yolov2/Detection";
  }

  static const char* value(const ::object_detection_yolov2::Detection_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::object_detection_yolov2::Detection_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n\
\n\
string object_class\n\
float32 p\n\
\n\
uint16 x\n\
uint16 y\n\
\n\
float32 cam_x\n\
float32 cam_y\n\
float32 cam_z\n\
\n\
uint16 width\n\
uint16 height\n\
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

  static const char* value(const ::object_detection_yolov2::Detection_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::object_detection_yolov2::Detection_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.object_class);
      stream.next(m.p);
      stream.next(m.x);
      stream.next(m.y);
      stream.next(m.cam_x);
      stream.next(m.cam_y);
      stream.next(m.cam_z);
      stream.next(m.width);
      stream.next(m.height);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Detection_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::object_detection_yolov2::Detection_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::object_detection_yolov2::Detection_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "object_class: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.object_class);
    s << indent << "p: ";
    Printer<float>::stream(s, indent + "  ", v.p);
    s << indent << "x: ";
    Printer<uint16_t>::stream(s, indent + "  ", v.x);
    s << indent << "y: ";
    Printer<uint16_t>::stream(s, indent + "  ", v.y);
    s << indent << "cam_x: ";
    Printer<float>::stream(s, indent + "  ", v.cam_x);
    s << indent << "cam_y: ";
    Printer<float>::stream(s, indent + "  ", v.cam_y);
    s << indent << "cam_z: ";
    Printer<float>::stream(s, indent + "  ", v.cam_z);
    s << indent << "width: ";
    Printer<uint16_t>::stream(s, indent + "  ", v.width);
    s << indent << "height: ";
    Printer<uint16_t>::stream(s, indent + "  ", v.height);
  }
};

} // namespace message_operations
} // namespace ros

#endif // OBJECT_DETECTION_YOLOV2_MESSAGE_DETECTION_H
