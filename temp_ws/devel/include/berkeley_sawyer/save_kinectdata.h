// Generated by gencpp from file berkeley_sawyer/save_kinectdata.msg
// DO NOT EDIT!


#ifndef BERKELEY_SAWYER_MESSAGE_SAVE_KINECTDATA_H
#define BERKELEY_SAWYER_MESSAGE_SAVE_KINECTDATA_H

#include <ros/service_traits.h>


#include <berkeley_sawyer/save_kinectdataRequest.h>
#include <berkeley_sawyer/save_kinectdataResponse.h>


namespace berkeley_sawyer
{

struct save_kinectdata
{

typedef save_kinectdataRequest Request;
typedef save_kinectdataResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct save_kinectdata
} // namespace berkeley_sawyer


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::berkeley_sawyer::save_kinectdata > {
  static const char* value()
  {
    return "36618b1eaec8d2483d42a27cb2744012";
  }

  static const char* value(const ::berkeley_sawyer::save_kinectdata&) { return value(); }
};

template<>
struct DataType< ::berkeley_sawyer::save_kinectdata > {
  static const char* value()
  {
    return "berkeley_sawyer/save_kinectdata";
  }

  static const char* value(const ::berkeley_sawyer::save_kinectdata&) { return value(); }
};


// service_traits::MD5Sum< ::berkeley_sawyer::save_kinectdataRequest> should match 
// service_traits::MD5Sum< ::berkeley_sawyer::save_kinectdata > 
template<>
struct MD5Sum< ::berkeley_sawyer::save_kinectdataRequest>
{
  static const char* value()
  {
    return MD5Sum< ::berkeley_sawyer::save_kinectdata >::value();
  }
  static const char* value(const ::berkeley_sawyer::save_kinectdataRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::berkeley_sawyer::save_kinectdataRequest> should match 
// service_traits::DataType< ::berkeley_sawyer::save_kinectdata > 
template<>
struct DataType< ::berkeley_sawyer::save_kinectdataRequest>
{
  static const char* value()
  {
    return DataType< ::berkeley_sawyer::save_kinectdata >::value();
  }
  static const char* value(const ::berkeley_sawyer::save_kinectdataRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::berkeley_sawyer::save_kinectdataResponse> should match 
// service_traits::MD5Sum< ::berkeley_sawyer::save_kinectdata > 
template<>
struct MD5Sum< ::berkeley_sawyer::save_kinectdataResponse>
{
  static const char* value()
  {
    return MD5Sum< ::berkeley_sawyer::save_kinectdata >::value();
  }
  static const char* value(const ::berkeley_sawyer::save_kinectdataResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::berkeley_sawyer::save_kinectdataResponse> should match 
// service_traits::DataType< ::berkeley_sawyer::save_kinectdata > 
template<>
struct DataType< ::berkeley_sawyer::save_kinectdataResponse>
{
  static const char* value()
  {
    return DataType< ::berkeley_sawyer::save_kinectdata >::value();
  }
  static const char* value(const ::berkeley_sawyer::save_kinectdataResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // BERKELEY_SAWYER_MESSAGE_SAVE_KINECTDATA_H
