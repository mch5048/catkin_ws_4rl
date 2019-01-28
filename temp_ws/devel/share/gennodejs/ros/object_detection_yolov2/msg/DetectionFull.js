// Auto-generated. Do not edit!

// (in-package object_detection_yolov2.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let DetectionArray = require('./DetectionArray.js');
let sensor_msgs = _finder('sensor_msgs');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class DetectionFull {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.image = null;
      this.masks = null;
      this.detections = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('image')) {
        this.image = initObj.image
      }
      else {
        this.image = new sensor_msgs.msg.Image();
      }
      if (initObj.hasOwnProperty('masks')) {
        this.masks = initObj.masks
      }
      else {
        this.masks = [];
      }
      if (initObj.hasOwnProperty('detections')) {
        this.detections = initObj.detections
      }
      else {
        this.detections = new DetectionArray();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type DetectionFull
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [image]
    bufferOffset = sensor_msgs.msg.Image.serialize(obj.image, buffer, bufferOffset);
    // Serialize message field [masks]
    // Serialize the length for message field [masks]
    bufferOffset = _serializer.uint32(obj.masks.length, buffer, bufferOffset);
    obj.masks.forEach((val) => {
      bufferOffset = sensor_msgs.msg.Image.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [detections]
    bufferOffset = DetectionArray.serialize(obj.detections, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type DetectionFull
    let len;
    let data = new DetectionFull(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [image]
    data.image = sensor_msgs.msg.Image.deserialize(buffer, bufferOffset);
    // Deserialize message field [masks]
    // Deserialize array length for message field [masks]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.masks = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.masks[i] = sensor_msgs.msg.Image.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [detections]
    data.detections = DetectionArray.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    length += sensor_msgs.msg.Image.getMessageSize(object.image);
    object.masks.forEach((val) => {
      length += sensor_msgs.msg.Image.getMessageSize(val);
    });
    length += DetectionArray.getMessageSize(object.detections);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'object_detection_yolov2/DetectionFull';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '3b39abf49a96981c609db709bdd09c4d';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    Header header
    
    # The image containing the detetions
    sensor_msgs/Image image
    
    # binary images containing masks
    sensor_msgs/Image[] masks
    
    # The array containing all the detections
    DetectionArray detections
    
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
    MSG: object_detection_yolov2/DetectionArray
    Header header
    
    # The size of the array
    uint32 size
    # The array containing all the detections
    Detection[] data
    
    ================================================================================
    MSG: object_detection_yolov2/Detection
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new DetectionFull(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.image !== undefined) {
      resolved.image = sensor_msgs.msg.Image.Resolve(msg.image)
    }
    else {
      resolved.image = new sensor_msgs.msg.Image()
    }

    if (msg.masks !== undefined) {
      resolved.masks = new Array(msg.masks.length);
      for (let i = 0; i < resolved.masks.length; ++i) {
        resolved.masks[i] = sensor_msgs.msg.Image.Resolve(msg.masks[i]);
      }
    }
    else {
      resolved.masks = []
    }

    if (msg.detections !== undefined) {
      resolved.detections = DetectionArray.Resolve(msg.detections)
    }
    else {
      resolved.detections = new DetectionArray()
    }

    return resolved;
    }
};

module.exports = DetectionFull;
