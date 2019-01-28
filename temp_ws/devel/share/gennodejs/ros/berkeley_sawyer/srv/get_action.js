// Auto-generated. Do not edit!

// (in-package berkeley_sawyer.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let sensor_msgs = _finder('sensor_msgs');

//-----------------------------------------------------------


//-----------------------------------------------------------

class get_actionRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.main = null;
      this.aux1 = null;
      this.state = null;
      this.desig_pos_aux1 = null;
      this.goal_pos_aux1 = null;
    }
    else {
      if (initObj.hasOwnProperty('main')) {
        this.main = initObj.main
      }
      else {
        this.main = new sensor_msgs.msg.Image();
      }
      if (initObj.hasOwnProperty('aux1')) {
        this.aux1 = initObj.aux1
      }
      else {
        this.aux1 = new sensor_msgs.msg.Image();
      }
      if (initObj.hasOwnProperty('state')) {
        this.state = initObj.state
      }
      else {
        this.state = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('desig_pos_aux1')) {
        this.desig_pos_aux1 = initObj.desig_pos_aux1
      }
      else {
        this.desig_pos_aux1 = new Array(4).fill(0);
      }
      if (initObj.hasOwnProperty('goal_pos_aux1')) {
        this.goal_pos_aux1 = initObj.goal_pos_aux1
      }
      else {
        this.goal_pos_aux1 = new Array(4).fill(0);
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type get_actionRequest
    // Serialize message field [main]
    bufferOffset = sensor_msgs.msg.Image.serialize(obj.main, buffer, bufferOffset);
    // Serialize message field [aux1]
    bufferOffset = sensor_msgs.msg.Image.serialize(obj.aux1, buffer, bufferOffset);
    // Check that the constant length array field [state] has the right length
    if (obj.state.length !== 3) {
      throw new Error('Unable to serialize array field state - length must be 3')
    }
    // Serialize message field [state]
    bufferOffset = _arraySerializer.float32(obj.state, buffer, bufferOffset, 3);
    // Check that the constant length array field [desig_pos_aux1] has the right length
    if (obj.desig_pos_aux1.length !== 4) {
      throw new Error('Unable to serialize array field desig_pos_aux1 - length must be 4')
    }
    // Serialize message field [desig_pos_aux1]
    bufferOffset = _arraySerializer.int64(obj.desig_pos_aux1, buffer, bufferOffset, 4);
    // Check that the constant length array field [goal_pos_aux1] has the right length
    if (obj.goal_pos_aux1.length !== 4) {
      throw new Error('Unable to serialize array field goal_pos_aux1 - length must be 4')
    }
    // Serialize message field [goal_pos_aux1]
    bufferOffset = _arraySerializer.int64(obj.goal_pos_aux1, buffer, bufferOffset, 4);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type get_actionRequest
    let len;
    let data = new get_actionRequest(null);
    // Deserialize message field [main]
    data.main = sensor_msgs.msg.Image.deserialize(buffer, bufferOffset);
    // Deserialize message field [aux1]
    data.aux1 = sensor_msgs.msg.Image.deserialize(buffer, bufferOffset);
    // Deserialize message field [state]
    data.state = _arrayDeserializer.float32(buffer, bufferOffset, 3)
    // Deserialize message field [desig_pos_aux1]
    data.desig_pos_aux1 = _arrayDeserializer.int64(buffer, bufferOffset, 4)
    // Deserialize message field [goal_pos_aux1]
    data.goal_pos_aux1 = _arrayDeserializer.int64(buffer, bufferOffset, 4)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += sensor_msgs.msg.Image.getMessageSize(object.main);
    length += sensor_msgs.msg.Image.getMessageSize(object.aux1);
    return length + 76;
  }

  static datatype() {
    // Returns string type for a service object
    return 'berkeley_sawyer/get_actionRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'ca94d53ef720720b393b0f064fba63f4';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    sensor_msgs/Image main
    sensor_msgs/Image aux1
    float32[3] state
    int64[4] desig_pos_aux1
    int64[4] goal_pos_aux1
    
    ================================================================================
    MSG: sensor_msgs/Image
    # This message contains an uncompressed image
    # (0, 0) is at top-left corner of image
    #
    
    Header header        # Header timestamp should be acquisition time of image
                         # Header frame_id should be optical frame of camera
                         # origin of frame should be optical center of cameara
                         # +x should point to the right in the image
                         # +y should point down in the image
                         # +z should point into to plane of the image
                         # If the frame_id here and the frame_id of the CameraInfo
                         # message associated with the image conflict
                         # the behavior is undefined
    
    uint32 height         # image height, that is, number of rows
    uint32 width          # image width, that is, number of columns
    
    # The legal values for encoding are in file src/image_encodings.cpp
    # If you want to standardize a new string format, join
    # ros-users@lists.sourceforge.net and send an email proposing a new encoding.
    
    string encoding       # Encoding of pixels -- channel meaning, ordering, size
                          # taken from the list of strings in include/sensor_msgs/image_encodings.h
    
    uint8 is_bigendian    # is this data bigendian?
    uint32 step           # Full row length in bytes
    uint8[] data          # actual matrix data, size is (step * rows)
    
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
    const resolved = new get_actionRequest(null);
    if (msg.main !== undefined) {
      resolved.main = sensor_msgs.msg.Image.Resolve(msg.main)
    }
    else {
      resolved.main = new sensor_msgs.msg.Image()
    }

    if (msg.aux1 !== undefined) {
      resolved.aux1 = sensor_msgs.msg.Image.Resolve(msg.aux1)
    }
    else {
      resolved.aux1 = new sensor_msgs.msg.Image()
    }

    if (msg.state !== undefined) {
      resolved.state = msg.state;
    }
    else {
      resolved.state = new Array(3).fill(0)
    }

    if (msg.desig_pos_aux1 !== undefined) {
      resolved.desig_pos_aux1 = msg.desig_pos_aux1;
    }
    else {
      resolved.desig_pos_aux1 = new Array(4).fill(0)
    }

    if (msg.goal_pos_aux1 !== undefined) {
      resolved.goal_pos_aux1 = msg.goal_pos_aux1;
    }
    else {
      resolved.goal_pos_aux1 = new Array(4).fill(0)
    }

    return resolved;
    }
};

class get_actionResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.action = null;
    }
    else {
      if (initObj.hasOwnProperty('action')) {
        this.action = initObj.action
      }
      else {
        this.action = new Array(4).fill(0);
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type get_actionResponse
    // Check that the constant length array field [action] has the right length
    if (obj.action.length !== 4) {
      throw new Error('Unable to serialize array field action - length must be 4')
    }
    // Serialize message field [action]
    bufferOffset = _arraySerializer.float32(obj.action, buffer, bufferOffset, 4);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type get_actionResponse
    let len;
    let data = new get_actionResponse(null);
    // Deserialize message field [action]
    data.action = _arrayDeserializer.float32(buffer, bufferOffset, 4)
    return data;
  }

  static getMessageSize(object) {
    return 16;
  }

  static datatype() {
    // Returns string type for a service object
    return 'berkeley_sawyer/get_actionResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'b938d6887d09f236ab99f56a97685191';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32[4] action
    
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new get_actionResponse(null);
    if (msg.action !== undefined) {
      resolved.action = msg.action;
    }
    else {
      resolved.action = new Array(4).fill(0)
    }

    return resolved;
    }
};

module.exports = {
  Request: get_actionRequest,
  Response: get_actionResponse,
  md5sum() { return 'ed42212579cedad52c84913bafdfccf2'; },
  datatype() { return 'berkeley_sawyer/get_action'; }
};
