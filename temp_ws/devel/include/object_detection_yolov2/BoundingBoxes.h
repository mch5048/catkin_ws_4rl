// Generated by gencpp from file object_detection_yolov2/BoundingBoxes.msg
// DO NOT EDIT!


#ifndef OBJECT_DETECTION_YOLOV2_MESSAGE_BOUNDINGBOXES_H
#define OBJECT_DETECTION_YOLOV2_MESSAGE_BOUNDINGBOXES_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <std_msgs/Header.h>
#include <object_detection_yolov2/BoundingBox.h>

namespace object_detection_yolov2
{
template <class ContainerAllocator>
struct BoundingBoxes_
{
  typedef BoundingBoxes_<ContainerAllocator> Type;

  BoundingBoxes_()
    : header()
    , image_header()
    , bounding_boxes()  {
    }
  BoundingBoxes_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , image_header(_alloc)
    , bounding_boxes(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef  ::std_msgs::Header_<ContainerAllocator>  _image_header_type;
  _image_header_type image_header;

   typedef std::vector< ::object_detection_yolov2::BoundingBox_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::object_detection_yolov2::BoundingBox_<ContainerAllocator> >::other >  _bounding_boxes_type;
  _bounding_boxes_type bounding_boxes;





  typedef boost::shared_ptr< ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> const> ConstPtr;

}; // struct BoundingBoxes_

typedef ::object_detection_yolov2::BoundingBoxes_<std::allocator<void> > BoundingBoxes;

typedef boost::shared_ptr< ::object_detection_yolov2::BoundingBoxes > BoundingBoxesPtr;
typedef boost::shared_ptr< ::object_detection_yolov2::BoundingBoxes const> BoundingBoxesConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> >::stream(s, "", v);
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
struct IsFixedSize< ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> >
{
  static const char* value()
  {
    return "c03e7499c2e5b938e301fea76459b092";
  }

  static const char* value(const ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xc03e7499c2e5b938ULL;
  static const uint64_t static_value2 = 0xe301fea76459b092ULL;
};

template<class ContainerAllocator>
struct DataType< ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> >
{
  static const char* value()
  {
    return "object_detection_yolov2/BoundingBoxes";
  }

  static const char* value(const ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n\
Header image_header\n\
BoundingBox[] bounding_boxes\n\
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
MSG: object_detection_yolov2/BoundingBox\n\
string Class\n\
float64 probability\n\
int64 xmin\n\
int64 ymin\n\
int64 xmax\n\
int64 ymax\n\
\n\
";
  }

  static const char* value(const ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.image_header);
      stream.next(m.bounding_boxes);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct BoundingBoxes_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::object_detection_yolov2::BoundingBoxes_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "image_header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.image_header);
    s << indent << "bounding_boxes[]" << std::endl;
    for (size_t i = 0; i < v.bounding_boxes.size(); ++i)
    {
      s << indent << "  bounding_boxes[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::object_detection_yolov2::BoundingBox_<ContainerAllocator> >::stream(s, indent + "    ", v.bounding_boxes[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // OBJECT_DETECTION_YOLOV2_MESSAGE_BOUNDINGBOXES_H
