// Generated by gencpp from file object_detection_yolov2/DetectionFull.msg
// DO NOT EDIT!


#ifndef OBJECT_DETECTION_YOLOV2_MESSAGE_DETECTIONFULL_H
#define OBJECT_DETECTION_YOLOV2_MESSAGE_DETECTIONFULL_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Image.h>
#include <object_detection_yolov2/DetectionArray.h>

namespace object_detection_yolov2
{
template <class ContainerAllocator>
struct DetectionFull_
{
  typedef DetectionFull_<ContainerAllocator> Type;

  DetectionFull_()
    : header()
    , image()
    , masks()
    , detections()  {
    }
  DetectionFull_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , image(_alloc)
    , masks(_alloc)
    , detections(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef  ::sensor_msgs::Image_<ContainerAllocator>  _image_type;
  _image_type image;

   typedef std::vector< ::sensor_msgs::Image_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::sensor_msgs::Image_<ContainerAllocator> >::other >  _masks_type;
  _masks_type masks;

   typedef  ::object_detection_yolov2::DetectionArray_<ContainerAllocator>  _detections_type;
  _detections_type detections;





  typedef boost::shared_ptr< ::object_detection_yolov2::DetectionFull_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::object_detection_yolov2::DetectionFull_<ContainerAllocator> const> ConstPtr;

}; // struct DetectionFull_

typedef ::object_detection_yolov2::DetectionFull_<std::allocator<void> > DetectionFull;

typedef boost::shared_ptr< ::object_detection_yolov2::DetectionFull > DetectionFullPtr;
typedef boost::shared_ptr< ::object_detection_yolov2::DetectionFull const> DetectionFullConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::object_detection_yolov2::DetectionFull_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::object_detection_yolov2::DetectionFull_<ContainerAllocator> >::stream(s, "", v);
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
struct IsFixedSize< ::object_detection_yolov2::DetectionFull_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::object_detection_yolov2::DetectionFull_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::object_detection_yolov2::DetectionFull_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::object_detection_yolov2::DetectionFull_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::object_detection_yolov2::DetectionFull_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::object_detection_yolov2::DetectionFull_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::object_detection_yolov2::DetectionFull_<ContainerAllocator> >
{
  static const char* value()
  {
    return "3b39abf49a96981c609db709bdd09c4d";
  }

  static const char* value(const ::object_detection_yolov2::DetectionFull_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x3b39abf49a96981cULL;
  static const uint64_t static_value2 = 0x609db709bdd09c4dULL;
};

template<class ContainerAllocator>
struct DataType< ::object_detection_yolov2::DetectionFull_<ContainerAllocator> >
{
  static const char* value()
  {
    return "object_detection_yolov2/DetectionFull";
  }

  static const char* value(const ::object_detection_yolov2::DetectionFull_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::object_detection_yolov2::DetectionFull_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n\
\n\
# The image containing the detetions\n\
sensor_msgs/Image image\n\
\n\
# binary images containing masks\n\
sensor_msgs/Image[] masks\n\
\n\
# The array containing all the detections\n\
DetectionArray detections\n\
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
MSG: sensor_msgs/Image\n\
# This message contains an uncompressed image\n\
# (0, 0) is at top-left corner of image\n\
#\n\
\n\
Header header        # Header timestamp should be acquisition time of image\n\
                     # Header frame_id should be optical frame of camera\n\
                     # origin of frame should be optical center of cameara\n\
                     # +x should point to the right in the image\n\
                     # +y should point down in the image\n\
                     # +z should point into to plane of the image\n\
                     # If the frame_id here and the frame_id of the CameraInfo\n\
                     # message associated with the image conflict\n\
                     # the behavior is undefined\n\
\n\
uint32 height         # image height, that is, number of rows\n\
uint32 width          # image width, that is, number of columns\n\
\n\
# The legal values for encoding are in file src/image_encodings.cpp\n\
# If you want to standardize a new string format, join\n\
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.\n\
\n\
string encoding       # Encoding of pixels -- channel meaning, ordering, size\n\
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h\n\
\n\
uint8 is_bigendian    # is this data bigendian?\n\
uint32 step           # Full row length in bytes\n\
uint8[] data          # actual matrix data, size is (step * rows)\n\
\n\
================================================================================\n\
MSG: object_detection_yolov2/DetectionArray\n\
Header header\n\
\n\
# The size of the array\n\
uint32 size\n\
# The array containing all the detections\n\
Detection[] data\n\
\n\
================================================================================\n\
MSG: object_detection_yolov2/Detection\n\
Header header\n\
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
";
  }

  static const char* value(const ::object_detection_yolov2::DetectionFull_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::object_detection_yolov2::DetectionFull_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.image);
      stream.next(m.masks);
      stream.next(m.detections);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct DetectionFull_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::object_detection_yolov2::DetectionFull_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::object_detection_yolov2::DetectionFull_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "image: ";
    s << std::endl;
    Printer< ::sensor_msgs::Image_<ContainerAllocator> >::stream(s, indent + "  ", v.image);
    s << indent << "masks[]" << std::endl;
    for (size_t i = 0; i < v.masks.size(); ++i)
    {
      s << indent << "  masks[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::sensor_msgs::Image_<ContainerAllocator> >::stream(s, indent + "    ", v.masks[i]);
    }
    s << indent << "detections: ";
    s << std::endl;
    Printer< ::object_detection_yolov2::DetectionArray_<ContainerAllocator> >::stream(s, indent + "  ", v.detections);
  }
};

} // namespace message_operations
} // namespace ros

#endif // OBJECT_DETECTION_YOLOV2_MESSAGE_DETECTIONFULL_H
