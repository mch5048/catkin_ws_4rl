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

class init_traj_visualmpcRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.itr = null;
      this.igrp = null;
      this.goalmain = null;
      this.goalaux1 = null;
      this.save_subdir = null;
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
      if (initObj.hasOwnProperty('goalmain')) {
        this.goalmain = initObj.goalmain
      }
      else {
        this.goalmain = new sensor_msgs.msg.Image();
      }
      if (initObj.hasOwnProperty('goalaux1')) {
        this.goalaux1 = initObj.goalaux1
      }
      else {
        this.goalaux1 = new sensor_msgs.msg.Image();
      }
      if (initObj.hasOwnProperty('save_subdir')) {
        this.save_subdir = initObj.save_subdir
      }
      else {
        this.save_subdir = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type init_traj_visualmpcRequest
    // Serialize message field [itr]
    bufferOffset = _serializer.int64(obj.itr, buffer, bufferOffset);
    // Serialize message field [igrp]
    bufferOffset = _serializer.int64(obj.igrp, buffer, bufferOffset);
    // Serialize message field [goalmain]
    bufferOffset = sensor_msgs.msg.Image.serialize(obj.goalmain, buffer, bufferOffset);
    // Serialize message field [goalaux1]
    bufferOffset = sensor_msgs.msg.Image.serialize(obj.goalaux1, buffer, bufferOffset);
    // Serialize message field [save_subdir]
    bufferOffset = _serializer.string(obj.save_subdir, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type init_traj_visualmpcRequest
    let len;
    let data = new init_traj_visualmpcRequest(null);
    // Deserialize message field [itr]
    data.itr = _deserializer.int64(buffer, bufferOffset);
    // Deserialize message field [igrp]
    data.igrp = _deserializer.int64(buffer, bufferOffset);
    // Deserialize message field [goalmain]
    data.goalmain = sensor_msgs.msg.Image.deserialize(buffer, bufferOffset);
    // Deserialize message field [goalaux1]
    data.goalaux1 = sensor_msgs.msg.Image.deserialize(buffer, bufferOffset);
    // Deserialize message field [save_subdir]
    data.save_subdir = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += sensor_msgs.msg.Image.getMessageSize(object.goalmain);
    length += sensor_msgs.msg.Image.getMessageSize(object.goalaux1);
    length += object.save_subdir.length;
    return length + 20;
  }

  static datatype() {
    // Returns string type for a service object
    return 'berkeley_sawyer/init_traj_visualmpcRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '212549b9c1a4ea535ff2ca3d14d779c5';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int64 itr
    int64 igrp
    sensor_msgs/Image goalmain
    sensor_msgs/Image goalaux1
    string save_subdir
    
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
    const resolved = new init_traj_visualmpcRequest(null);
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

    if (msg.goalmain !== undefined) {
      resolved.goalmain = sensor_msgs.msg.Image.Resolve(msg.goalmain)
    }
    else {
      resolved.goalmain = new sensor_msgs.msg.Image()
    }

    if (msg.goalaux1 !== undefined) {
      resolved.goalaux1 = sensor_msgs.msg.Image.Resolve(msg.goalaux1)
    }
    else {
      resolved.goalaux1 = new sensor_msgs.msg.Image()
    }

    if (msg.save_subdir !== undefined) {
      resolved.save_subdir = msg.save_subdir;
    }
    else {
      resolved.save_subdir = ''
    }

    return resolved;
    }
};

class init_traj_visualmpcResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type init_traj_visualmpcResponse
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type init_traj_visualmpcResponse
    let len;
    let data = new init_traj_visualmpcResponse(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'berkeley_sawyer/init_traj_visualmpcResponse';
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
    const resolved = new init_traj_visualmpcResponse(null);
    return resolved;
    }
};

module.exports = {
  Request: init_traj_visualmpcRequest,
  Response: init_traj_visualmpcResponse,
  md5sum() { return '212549b9c1a4ea535ff2ca3d14d779c5'; },
  datatype() { return 'berkeley_sawyer/init_traj_visualmpc'; }
};
