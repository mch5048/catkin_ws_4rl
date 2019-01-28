// Auto-generated. Do not edit!

// (in-package gazebo_ext_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class GetSkyPropertiesRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetSkyPropertiesRequest
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetSkyPropertiesRequest
    let len;
    let data = new GetSkyPropertiesRequest(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_ext_msgs/GetSkyPropertiesRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd41d8cd98f00b204e9800998ecf8427e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetSkyPropertiesRequest(null);
    return resolved;
    }
};

class GetSkyPropertiesResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.time = null;
      this.sunrise = null;
      this.sunset = null;
      this.wind_speed = null;
      this.wind_direction = null;
      this.cloud_ambient = null;
      this.humidity = null;
      this.mean_cloud_size = null;
      this.success = null;
      this.status_message = null;
    }
    else {
      if (initObj.hasOwnProperty('time')) {
        this.time = initObj.time
      }
      else {
        this.time = 0.0;
      }
      if (initObj.hasOwnProperty('sunrise')) {
        this.sunrise = initObj.sunrise
      }
      else {
        this.sunrise = 0.0;
      }
      if (initObj.hasOwnProperty('sunset')) {
        this.sunset = initObj.sunset
      }
      else {
        this.sunset = 0.0;
      }
      if (initObj.hasOwnProperty('wind_speed')) {
        this.wind_speed = initObj.wind_speed
      }
      else {
        this.wind_speed = 0.0;
      }
      if (initObj.hasOwnProperty('wind_direction')) {
        this.wind_direction = initObj.wind_direction
      }
      else {
        this.wind_direction = 0.0;
      }
      if (initObj.hasOwnProperty('cloud_ambient')) {
        this.cloud_ambient = initObj.cloud_ambient
      }
      else {
        this.cloud_ambient = new std_msgs.msg.ColorRGBA();
      }
      if (initObj.hasOwnProperty('humidity')) {
        this.humidity = initObj.humidity
      }
      else {
        this.humidity = 0.0;
      }
      if (initObj.hasOwnProperty('mean_cloud_size')) {
        this.mean_cloud_size = initObj.mean_cloud_size
      }
      else {
        this.mean_cloud_size = 0.0;
      }
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
      if (initObj.hasOwnProperty('status_message')) {
        this.status_message = initObj.status_message
      }
      else {
        this.status_message = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetSkyPropertiesResponse
    // Serialize message field [time]
    bufferOffset = _serializer.float64(obj.time, buffer, bufferOffset);
    // Serialize message field [sunrise]
    bufferOffset = _serializer.float64(obj.sunrise, buffer, bufferOffset);
    // Serialize message field [sunset]
    bufferOffset = _serializer.float64(obj.sunset, buffer, bufferOffset);
    // Serialize message field [wind_speed]
    bufferOffset = _serializer.float64(obj.wind_speed, buffer, bufferOffset);
    // Serialize message field [wind_direction]
    bufferOffset = _serializer.float64(obj.wind_direction, buffer, bufferOffset);
    // Serialize message field [cloud_ambient]
    bufferOffset = std_msgs.msg.ColorRGBA.serialize(obj.cloud_ambient, buffer, bufferOffset);
    // Serialize message field [humidity]
    bufferOffset = _serializer.float64(obj.humidity, buffer, bufferOffset);
    // Serialize message field [mean_cloud_size]
    bufferOffset = _serializer.float64(obj.mean_cloud_size, buffer, bufferOffset);
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    // Serialize message field [status_message]
    bufferOffset = _serializer.string(obj.status_message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetSkyPropertiesResponse
    let len;
    let data = new GetSkyPropertiesResponse(null);
    // Deserialize message field [time]
    data.time = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [sunrise]
    data.sunrise = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [sunset]
    data.sunset = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [wind_speed]
    data.wind_speed = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [wind_direction]
    data.wind_direction = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [cloud_ambient]
    data.cloud_ambient = std_msgs.msg.ColorRGBA.deserialize(buffer, bufferOffset);
    // Deserialize message field [humidity]
    data.humidity = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [mean_cloud_size]
    data.mean_cloud_size = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [status_message]
    data.status_message = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += object.status_message.length;
    return length + 77;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_ext_msgs/GetSkyPropertiesResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '785496b76654a68787d4a95f6a2d2af4';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float64 time
    float64 sunrise
    float64 sunset
    float64 wind_speed
    float64 wind_direction
    std_msgs/ColorRGBA cloud_ambient
    float64 humidity
    float64 mean_cloud_size
    bool success
    string status_message
    
    
    ================================================================================
    MSG: std_msgs/ColorRGBA
    float32 r
    float32 g
    float32 b
    float32 a
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetSkyPropertiesResponse(null);
    if (msg.time !== undefined) {
      resolved.time = msg.time;
    }
    else {
      resolved.time = 0.0
    }

    if (msg.sunrise !== undefined) {
      resolved.sunrise = msg.sunrise;
    }
    else {
      resolved.sunrise = 0.0
    }

    if (msg.sunset !== undefined) {
      resolved.sunset = msg.sunset;
    }
    else {
      resolved.sunset = 0.0
    }

    if (msg.wind_speed !== undefined) {
      resolved.wind_speed = msg.wind_speed;
    }
    else {
      resolved.wind_speed = 0.0
    }

    if (msg.wind_direction !== undefined) {
      resolved.wind_direction = msg.wind_direction;
    }
    else {
      resolved.wind_direction = 0.0
    }

    if (msg.cloud_ambient !== undefined) {
      resolved.cloud_ambient = std_msgs.msg.ColorRGBA.Resolve(msg.cloud_ambient)
    }
    else {
      resolved.cloud_ambient = new std_msgs.msg.ColorRGBA()
    }

    if (msg.humidity !== undefined) {
      resolved.humidity = msg.humidity;
    }
    else {
      resolved.humidity = 0.0
    }

    if (msg.mean_cloud_size !== undefined) {
      resolved.mean_cloud_size = msg.mean_cloud_size;
    }
    else {
      resolved.mean_cloud_size = 0.0
    }

    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    if (msg.status_message !== undefined) {
      resolved.status_message = msg.status_message;
    }
    else {
      resolved.status_message = ''
    }

    return resolved;
    }
};

module.exports = {
  Request: GetSkyPropertiesRequest,
  Response: GetSkyPropertiesResponse,
  md5sum() { return '785496b76654a68787d4a95f6a2d2af4'; },
  datatype() { return 'gazebo_ext_msgs/GetSkyProperties'; }
};
