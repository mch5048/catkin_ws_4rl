// Generated by gencpp from file berkeley_sawyer/get_actionRequest.msg
// DO NOT EDIT!


#ifndef BERKELEY_SAWYER_MESSAGE_GET_ACTIONREQUEST_H
#define BERKELEY_SAWYER_MESSAGE_GET_ACTIONREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/Image.h>

namespace berkeley_sawyer
{
template <class ContainerAllocator>
struct get_actionRequest_
{
  typedef get_actionRequest_<ContainerAllocator> Type;

  get_actionRequest_()
    : main()
    , aux1()
    , state()
    , desig_pos_aux1()
    , goal_pos_aux1()  {
      state.assign(0.0);

      desig_pos_aux1.assign(0);

      goal_pos_aux1.assign(0);
  }
  get_actionRequest_(const ContainerAllocator& _alloc)
    : main(_alloc)
    , aux1(_alloc)
    , state()
    , desig_pos_aux1()
    , goal_pos_aux1()  {
  (void)_alloc;
      state.assign(0.0);

      desig_pos_aux1.assign(0);

      goal_pos_aux1.assign(0);
  }



   typedef  ::sensor_msgs::Image_<ContainerAllocator>  _main_type;
  _main_type main;

   typedef  ::sensor_msgs::Image_<ContainerAllocator>  _aux1_type;
  _aux1_type aux1;

   typedef boost::array<float, 3>  _state_type;
  _state_type state;

   typedef boost::array<int64_t, 4>  _desig_pos_aux1_type;
  _desig_pos_aux1_type desig_pos_aux1;

   typedef boost::array<int64_t, 4>  _goal_pos_aux1_type;
  _goal_pos_aux1_type goal_pos_aux1;





  typedef boost::shared_ptr< ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> const> ConstPtr;

}; // struct get_actionRequest_

typedef ::berkeley_sawyer::get_actionRequest_<std::allocator<void> > get_actionRequest;

typedef boost::shared_ptr< ::berkeley_sawyer::get_actionRequest > get_actionRequestPtr;
typedef boost::shared_ptr< ::berkeley_sawyer::get_actionRequest const> get_actionRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace berkeley_sawyer

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ca94d53ef720720b393b0f064fba63f4";
  }

  static const char* value(const ::berkeley_sawyer::get_actionRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xca94d53ef720720bULL;
  static const uint64_t static_value2 = 0x393b0f064fba63f4ULL;
};

template<class ContainerAllocator>
struct DataType< ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "berkeley_sawyer/get_actionRequest";
  }

  static const char* value(const ::berkeley_sawyer::get_actionRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "sensor_msgs/Image main\n\
sensor_msgs/Image aux1\n\
float32[3] state\n\
int64[4] desig_pos_aux1\n\
int64[4] goal_pos_aux1\n\
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

  static const char* value(const ::berkeley_sawyer::get_actionRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.main);
      stream.next(m.aux1);
      stream.next(m.state);
      stream.next(m.desig_pos_aux1);
      stream.next(m.goal_pos_aux1);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct get_actionRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::berkeley_sawyer::get_actionRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::berkeley_sawyer::get_actionRequest_<ContainerAllocator>& v)
  {
    s << indent << "main: ";
    s << std::endl;
    Printer< ::sensor_msgs::Image_<ContainerAllocator> >::stream(s, indent + "  ", v.main);
    s << indent << "aux1: ";
    s << std::endl;
    Printer< ::sensor_msgs::Image_<ContainerAllocator> >::stream(s, indent + "  ", v.aux1);
    s << indent << "state[]" << std::endl;
    for (size_t i = 0; i < v.state.size(); ++i)
    {
      s << indent << "  state[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.state[i]);
    }
    s << indent << "desig_pos_aux1[]" << std::endl;
    for (size_t i = 0; i < v.desig_pos_aux1.size(); ++i)
    {
      s << indent << "  desig_pos_aux1[" << i << "]: ";
      Printer<int64_t>::stream(s, indent + "  ", v.desig_pos_aux1[i]);
    }
    s << indent << "goal_pos_aux1[]" << std::endl;
    for (size_t i = 0; i < v.goal_pos_aux1.size(); ++i)
    {
      s << indent << "  goal_pos_aux1[" << i << "]: ";
      Printer<int64_t>::stream(s, indent + "  ", v.goal_pos_aux1[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // BERKELEY_SAWYER_MESSAGE_GET_ACTIONREQUEST_H
