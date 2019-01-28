// Auto-generated. Do not edit!

// (in-package object_detection_yolov2.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class Detection {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.object_class = null;
      this.p = null;
      this.x = null;
      this.y = null;
      this.cam_x = null;
      this.cam_y = null;
      this.cam_z = null;
      this.width = null;
      this.height = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('object_class')) {
        this.object_class = initObj.object_class
      }
      else {
        this.object_class = '';
      }
      if (initObj.hasOwnProperty('p')) {
        this.p = initObj.p
      }
      else {
        this.p = 0.0;
      }
      if (initObj.hasOwnProperty('x')) {
        this.x = initObj.x
      }
      else {
        this.x = 0;
      }
      if (initObj.hasOwnProperty('y')) {
        this.y = initObj.y
      }
      else {
        this.y = 0;
      }
      if (initObj.hasOwnProperty('cam_x')) {
        this.cam_x = initObj.cam_x
      }
      else {
        this.cam_x = 0.0;
      }
      if (initObj.hasOwnProperty('cam_y')) {
        this.cam_y = initObj.cam_y
      }
      else {
        this.cam_y = 0.0;
      }
      if (initObj.hasOwnProperty('cam_z')) {
        this.cam_z = initObj.cam_z
      }
      else {
        this.cam_z = 0.0;
      }
      if (initObj.hasOwnProperty('width')) {
        this.width = initObj.width
      }
      else {
        this.width = 0;
      }
      if (initObj.hasOwnProperty('height')) {
        this.height = initObj.height
      }
      else {
        this.height = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Detection
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [object_class]
    bufferOffset = _serializer.string(obj.object_class, buffer, bufferOffset);
    // Serialize message field [p]
    bufferOffset = _serializer.float32(obj.p, buffer, bufferOffset);
    // Serialize message field [x]
    bufferOffset = _serializer.uint16(obj.x, buffer, bufferOffset);
    // Serialize message field [y]
    bufferOffset = _serializer.uint16(obj.y, buffer, bufferOffset);
    // Serialize message field [cam_x]
    bufferOffset = _serializer.float32(obj.cam_x, buffer, bufferOffset);
    // Serialize message field [cam_y]
    bufferOffset = _serializer.float32(obj.cam_y, buffer, bufferOffset);
    // Serialize message field [cam_z]
    bufferOffset = _serializer.float32(obj.cam_z, buffer, bufferOffset);
    // Serialize message field [width]
    bufferOffset = _serializer.uint16(obj.width, buffer, bufferOffset);
    // Serialize message field [height]
    bufferOffset = _serializer.uint16(obj.height, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Detection
    let len;
    let data = new Detection(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [object_class]
    data.object_class = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [p]
    data.p = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [x]
    data.x = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [y]
    data.y = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [cam_x]
    data.cam_x = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [cam_y]
    data.cam_y = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [cam_z]
    data.cam_z = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [width]
    data.width = _deserializer.uint16(buffer, bufferOffset);
    // Deserialize message field [height]
    data.height = _deserializer.uint16(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += object.object_class.length;
    return length + 28;
  }

  static datatype() {
    // Returns string type for a message object
    return 'object_detection_yolov2/Detection';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd82132341465f8c6318faea203b0884c';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    
    string object_class
    float32 p
    
    uint16 x
    uint16 y
    
    float32 cam_x
    float32 cam_y
    float32 cam_z
    
    uint16 width
    uint16 height
    
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    # 0: no frame
    # 1: global frame
    string frame_id
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Detection(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.object_class !== undefined) {
      resolved.object_class = msg.object_class;
    }
    else {
      resolved.object_class = ''
    }

    if (msg.p !== undefined) {
      resolved.p = msg.p;
    }
    else {
      resolved.p = 0.0
    }

    if (msg.x !== undefined) {
      resolved.x = msg.x;
    }
    else {
      resolved.x = 0
    }

    if (msg.y !== undefined) {
      resolved.y = msg.y;
    }
    else {
      resolved.y = 0
    }

    if (msg.cam_x !== undefined) {
      resolved.cam_x = msg.cam_x;
    }
    else {
      resolved.cam_x = 0.0
    }

    if (msg.cam_y !== undefined) {
      resolved.cam_y = msg.cam_y;
    }
    else {
      resolved.cam_y = 0.0
    }

    if (msg.cam_z !== undefined) {
      resolved.cam_z = msg.cam_z;
    }
    else {
      resolved.cam_z = 0.0
    }

    if (msg.width !== undefined) {
      resolved.width = msg.width;
    }
    else {
      resolved.width = 0
    }

    if (msg.height !== undefined) {
      resolved.height = msg.height;
    }
    else {
      resolved.height = 0
    }

    return resolved;
    }
};

module.exports = Detection;
