// Auto-generated. Do not edit!

// (in-package rospy_message_converter.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class Uint8Array3TestMessage {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.data = null;
    }
    else {
      if (initObj.hasOwnProperty('data')) {
        this.data = initObj.data
      }
      else {
        this.data = new Array(3).fill(0);
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Uint8Array3TestMessage
    // Check that the constant length array field [data] has the right length
    if (obj.data.length !== 3) {
      throw new Error('Unable to serialize array field data - length must be 3')
    }
    // Serialize message field [data]
    bufferOffset = _arraySerializer.uint8(obj.data, buffer, bufferOffset, 3);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Uint8Array3TestMessage
    let len;
    let data = new Uint8Array3TestMessage(null);
    // Deserialize message field [data]
    data.data = _arrayDeserializer.uint8(buffer, bufferOffset, 3)
    return data;
  }

  static getMessageSize(object) {
    return 3;
  }

  static datatype() {
    // Returns string type for a message object
    return 'rospy_message_converter/Uint8Array3TestMessage';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '0aff379edf54a041a83af067122e7bc6';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Fixed size uint8 array for testing purposes
    uint8[3] data
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Uint8Array3TestMessage(null);
    if (msg.data !== undefined) {
      resolved.data = msg.data;
    }
    else {
      resolved.data = new Array(3).fill(0)
    }

    return resolved;
    }
};

module.exports = Uint8Array3TestMessage;
