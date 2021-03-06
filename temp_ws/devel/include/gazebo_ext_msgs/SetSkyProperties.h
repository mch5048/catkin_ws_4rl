// Generated by gencpp from file gazebo_ext_msgs/SetSkyProperties.msg
// DO NOT EDIT!


#ifndef GAZEBO_EXT_MSGS_MESSAGE_SETSKYPROPERTIES_H
#define GAZEBO_EXT_MSGS_MESSAGE_SETSKYPROPERTIES_H

#include <ros/service_traits.h>


#include <gazebo_ext_msgs/SetSkyPropertiesRequest.h>
#include <gazebo_ext_msgs/SetSkyPropertiesResponse.h>


namespace gazebo_ext_msgs
{

struct SetSkyProperties
{

typedef SetSkyPropertiesRequest Request;
typedef SetSkyPropertiesResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct SetSkyProperties
} // namespace gazebo_ext_msgs


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::gazebo_ext_msgs::SetSkyProperties > {
  static const char* value()
  {
    return "58ab80a272655f0012daa4ccd6a59539";
  }

  static const char* value(const ::gazebo_ext_msgs::SetSkyProperties&) { return value(); }
};

template<>
struct DataType< ::gazebo_ext_msgs::SetSkyProperties > {
  static const char* value()
  {
    return "gazebo_ext_msgs/SetSkyProperties";
  }

  static const char* value(const ::gazebo_ext_msgs::SetSkyProperties&) { return value(); }
};


// service_traits::MD5Sum< ::gazebo_ext_msgs::SetSkyPropertiesRequest> should match 
// service_traits::MD5Sum< ::gazebo_ext_msgs::SetSkyProperties > 
template<>
struct MD5Sum< ::gazebo_ext_msgs::SetSkyPropertiesRequest>
{
  static const char* value()
  {
    return MD5Sum< ::gazebo_ext_msgs::SetSkyProperties >::value();
  }
  static const char* value(const ::gazebo_ext_msgs::SetSkyPropertiesRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::gazebo_ext_msgs::SetSkyPropertiesRequest> should match 
// service_traits::DataType< ::gazebo_ext_msgs::SetSkyProperties > 
template<>
struct DataType< ::gazebo_ext_msgs::SetSkyPropertiesRequest>
{
  static const char* value()
  {
    return DataType< ::gazebo_ext_msgs::SetSkyProperties >::value();
  }
  static const char* value(const ::gazebo_ext_msgs::SetSkyPropertiesRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::gazebo_ext_msgs::SetSkyPropertiesResponse> should match 
// service_traits::MD5Sum< ::gazebo_ext_msgs::SetSkyProperties > 
template<>
struct MD5Sum< ::gazebo_ext_msgs::SetSkyPropertiesResponse>
{
  static const char* value()
  {
    return MD5Sum< ::gazebo_ext_msgs::SetSkyProperties >::value();
  }
  static const char* value(const ::gazebo_ext_msgs::SetSkyPropertiesResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::gazebo_ext_msgs::SetSkyPropertiesResponse> should match 
// service_traits::DataType< ::gazebo_ext_msgs::SetSkyProperties > 
template<>
struct DataType< ::gazebo_ext_msgs::SetSkyPropertiesResponse>
{
  static const char* value()
  {
    return DataType< ::gazebo_ext_msgs::SetSkyProperties >::value();
  }
  static const char* value(const ::gazebo_ext_msgs::SetSkyPropertiesResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // GAZEBO_EXT_MSGS_MESSAGE_SETSKYPROPERTIES_H
