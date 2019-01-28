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


//-----------------------------------------------------------

class GetCollisionNamesRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.link_names = null;
    }
    else {
      if (initObj.hasOwnProperty('link_names')) {
        this.link_names = initObj.link_names
      }
      else {
        this.link_names = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetCollisionNamesRequest
    // Serialize message field [link_names]
    bufferOffset = _arraySerializer.string(obj.link_names, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetCollisionNamesRequest
    let len;
    let data = new GetCollisionNamesRequest(null);
    // Deserialize message field [link_names]
    data.link_names = _arrayDeserializer.string(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.link_names.forEach((val) => {
      length += 4 + val.length;
    });
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_ext_msgs/GetCollisionNamesRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '8f2ac94008b559adc87f7d99565ba995';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string[] link_names
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetCollisionNamesRequest(null);
    if (msg.link_names !== undefined) {
      resolved.link_names = msg.link_names;
    }
    else {
      resolved.link_names = []
    }

    return resolved;
    }
};

class GetCollisionNamesResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.link_collision_names = null;
      this.success = null;
      this.status_message = null;
    }
    else {
      if (initObj.hasOwnProperty('link_collision_names')) {
        this.link_collision_names = initObj.link_collision_names
      }
      else {
        this.link_collision_names = [];
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
    // Serializes a message object of type GetCollisionNamesResponse
    // Serialize message field [link_collision_names]
    bufferOffset = _arraySerializer.string(obj.link_collision_names, buffer, bufferOffset, null);
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    // Serialize message field [status_message]
    bufferOffset = _serializer.string(obj.status_message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetCollisionNamesResponse
    let len;
    let data = new GetCollisionNamesResponse(null);
    // Deserialize message field [link_collision_names]
    data.link_collision_names = _arrayDeserializer.string(buffer, bufferOffset, null)
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [status_message]
    data.status_message = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.link_collision_names.forEach((val) => {
      length += 4 + val.length;
    });
    length += object.status_message.length;
    return length + 9;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_ext_msgs/GetCollisionNamesResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '68513360f32cd5fa3f85838fe35ea77d';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string[] link_collision_names
    bool success
    string status_message
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetCollisionNamesResponse(null);
    if (msg.link_collision_names !== undefined) {
      resolved.link_collision_names = msg.link_collision_names;
    }
    else {
      resolved.link_collision_names = []
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
  Request: GetCollisionNamesRequest,
  Response: GetCollisionNamesResponse,
  md5sum() { return '440bd39ad26bc774c2ef6bcfe06d56ea'; },
  datatype() { return 'gazebo_ext_msgs/GetCollisionNames'; }
};
