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

class GetVisualNamesRequest {
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
    // Serializes a message object of type GetVisualNamesRequest
    // Serialize message field [link_names]
    bufferOffset = _arraySerializer.string(obj.link_names, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetVisualNamesRequest
    let len;
    let data = new GetVisualNamesRequest(null);
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
    return 'gazebo_ext_msgs/GetVisualNamesRequest';
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
    const resolved = new GetVisualNamesRequest(null);
    if (msg.link_names !== undefined) {
      resolved.link_names = msg.link_names;
    }
    else {
      resolved.link_names = []
    }

    return resolved;
    }
};

class GetVisualNamesResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.link_visual_names = null;
      this.success = null;
      this.status_message = null;
    }
    else {
      if (initObj.hasOwnProperty('link_visual_names')) {
        this.link_visual_names = initObj.link_visual_names
      }
      else {
        this.link_visual_names = [];
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
    // Serializes a message object of type GetVisualNamesResponse
    // Serialize message field [link_visual_names]
    bufferOffset = _arraySerializer.string(obj.link_visual_names, buffer, bufferOffset, null);
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    // Serialize message field [status_message]
    bufferOffset = _serializer.string(obj.status_message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetVisualNamesResponse
    let len;
    let data = new GetVisualNamesResponse(null);
    // Deserialize message field [link_visual_names]
    data.link_visual_names = _arrayDeserializer.string(buffer, bufferOffset, null)
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [status_message]
    data.status_message = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.link_visual_names.forEach((val) => {
      length += 4 + val.length;
    });
    length += object.status_message.length;
    return length + 9;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_ext_msgs/GetVisualNamesResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '15672b4bf3481d17f0346525146875cb';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string[] link_visual_names
    bool success
    string status_message
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetVisualNamesResponse(null);
    if (msg.link_visual_names !== undefined) {
      resolved.link_visual_names = msg.link_visual_names;
    }
    else {
      resolved.link_visual_names = []
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
  Request: GetVisualNamesRequest,
  Response: GetVisualNamesResponse,
  md5sum() { return '84bee8e2f694c3877c28343b65184d47'; },
  datatype() { return 'gazebo_ext_msgs/GetVisualNames'; }
};
