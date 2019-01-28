// Auto-generated. Do not edit!

// (in-package berkeley_sawyer.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class save_kinectdataRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.itr = null;
    }
    else {
      if (initObj.hasOwnProperty('itr')) {
        this.itr = initObj.itr
      }
      else {
        this.itr = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type save_kinectdataRequest
    // Serialize message field [itr]
    bufferOffset = _serializer.int64(obj.itr, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type save_kinectdataRequest
    let len;
    let data = new save_kinectdataRequest(null);
    // Deserialize message field [itr]
    data.itr = _deserializer.int64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'berkeley_sawyer/save_kinectdataRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '36618b1eaec8d2483d42a27cb2744012';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int64 itr
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new save_kinectdataRequest(null);
    if (msg.itr !== undefined) {
      resolved.itr = msg.itr;
    }
    else {
      resolved.itr = 0
    }

    return resolved;
    }
};

class save_kinectdataResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type save_kinectdataResponse
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type save_kinectdataResponse
    let len;
    let data = new save_kinectdataResponse(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'berkeley_sawyer/save_kinectdataResponse';
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
    const resolved = new save_kinectdataResponse(null);
    return resolved;
    }
};

module.exports = {
  Request: save_kinectdataRequest,
  Response: save_kinectdataResponse,
  md5sum() { return '36618b1eaec8d2483d42a27cb2744012'; },
  datatype() { return 'berkeley_sawyer/save_kinectdata'; }
};
