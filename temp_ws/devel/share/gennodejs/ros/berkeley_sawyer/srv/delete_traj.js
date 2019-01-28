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

class delete_trajRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.itr = null;
      this.igrp = null;
    }
    else {
      if (initObj.hasOwnProperty('itr')) {
        this.itr = initObj.itr
      }
      else {
        this.itr = 0;
      }
      if (initObj.hasOwnProperty('igrp')) {
        this.igrp = initObj.igrp
      }
      else {
        this.igrp = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type delete_trajRequest
    // Serialize message field [itr]
    bufferOffset = _serializer.int64(obj.itr, buffer, bufferOffset);
    // Serialize message field [igrp]
    bufferOffset = _serializer.int64(obj.igrp, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type delete_trajRequest
    let len;
    let data = new delete_trajRequest(null);
    // Deserialize message field [itr]
    data.itr = _deserializer.int64(buffer, bufferOffset);
    // Deserialize message field [igrp]
    data.igrp = _deserializer.int64(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 16;
  }

  static datatype() {
    // Returns string type for a service object
    return 'berkeley_sawyer/delete_trajRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '58b1b06ba616229bc38a06a0a0af5730';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int64 itr
    int64 igrp
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new delete_trajRequest(null);
    if (msg.itr !== undefined) {
      resolved.itr = msg.itr;
    }
    else {
      resolved.itr = 0
    }

    if (msg.igrp !== undefined) {
      resolved.igrp = msg.igrp;
    }
    else {
      resolved.igrp = 0
    }

    return resolved;
    }
};

class delete_trajResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type delete_trajResponse
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type delete_trajResponse
    let len;
    let data = new delete_trajResponse(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'berkeley_sawyer/delete_trajResponse';
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
    const resolved = new delete_trajResponse(null);
    return resolved;
    }
};

module.exports = {
  Request: delete_trajRequest,
  Response: delete_trajResponse,
  md5sum() { return '58b1b06ba616229bc38a06a0a0af5730'; },
  datatype() { return 'berkeley_sawyer/delete_traj'; }
};
