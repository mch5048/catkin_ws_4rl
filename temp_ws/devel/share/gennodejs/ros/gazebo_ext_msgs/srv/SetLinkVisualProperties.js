// Auto-generated. Do not edit!

// (in-package gazebo_ext_msgs.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------


//-----------------------------------------------------------

class SetLinkVisualPropertiesRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.link_visual_name = null;
      this.ambient = null;
      this.diffuse = null;
      this.specular = null;
      this.emissive = null;
    }
    else {
      if (initObj.hasOwnProperty('link_visual_name')) {
        this.link_visual_name = initObj.link_visual_name
      }
      else {
        this.link_visual_name = '';
      }
      if (initObj.hasOwnProperty('ambient')) {
        this.ambient = initObj.ambient
      }
      else {
        this.ambient = new std_msgs.msg.ColorRGBA();
      }
      if (initObj.hasOwnProperty('diffuse')) {
        this.diffuse = initObj.diffuse
      }
      else {
        this.diffuse = new std_msgs.msg.ColorRGBA();
      }
      if (initObj.hasOwnProperty('specular')) {
        this.specular = initObj.specular
      }
      else {
        this.specular = new std_msgs.msg.ColorRGBA();
      }
      if (initObj.hasOwnProperty('emissive')) {
        this.emissive = initObj.emissive
      }
      else {
        this.emissive = new std_msgs.msg.ColorRGBA();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetLinkVisualPropertiesRequest
    // Serialize message field [link_visual_name]
    bufferOffset = _serializer.string(obj.link_visual_name, buffer, bufferOffset);
    // Serialize message field [ambient]
    bufferOffset = std_msgs.msg.ColorRGBA.serialize(obj.ambient, buffer, bufferOffset);
    // Serialize message field [diffuse]
    bufferOffset = std_msgs.msg.ColorRGBA.serialize(obj.diffuse, buffer, bufferOffset);
    // Serialize message field [specular]
    bufferOffset = std_msgs.msg.ColorRGBA.serialize(obj.specular, buffer, bufferOffset);
    // Serialize message field [emissive]
    bufferOffset = std_msgs.msg.ColorRGBA.serialize(obj.emissive, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetLinkVisualPropertiesRequest
    let len;
    let data = new SetLinkVisualPropertiesRequest(null);
    // Deserialize message field [link_visual_name]
    data.link_visual_name = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [ambient]
    data.ambient = std_msgs.msg.ColorRGBA.deserialize(buffer, bufferOffset);
    // Deserialize message field [diffuse]
    data.diffuse = std_msgs.msg.ColorRGBA.deserialize(buffer, bufferOffset);
    // Deserialize message field [specular]
    data.specular = std_msgs.msg.ColorRGBA.deserialize(buffer, bufferOffset);
    // Deserialize message field [emissive]
    data.emissive = std_msgs.msg.ColorRGBA.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += object.link_visual_name.length;
    return length + 68;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_ext_msgs/SetLinkVisualPropertiesRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '3d2c0a1301a984c848faa1e2d725f0cf';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string link_visual_name
    std_msgs/ColorRGBA ambient
    std_msgs/ColorRGBA diffuse
    std_msgs/ColorRGBA specular
    std_msgs/ColorRGBA emissive
    
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
    const resolved = new SetLinkVisualPropertiesRequest(null);
    if (msg.link_visual_name !== undefined) {
      resolved.link_visual_name = msg.link_visual_name;
    }
    else {
      resolved.link_visual_name = ''
    }

    if (msg.ambient !== undefined) {
      resolved.ambient = std_msgs.msg.ColorRGBA.Resolve(msg.ambient)
    }
    else {
      resolved.ambient = new std_msgs.msg.ColorRGBA()
    }

    if (msg.diffuse !== undefined) {
      resolved.diffuse = std_msgs.msg.ColorRGBA.Resolve(msg.diffuse)
    }
    else {
      resolved.diffuse = new std_msgs.msg.ColorRGBA()
    }

    if (msg.specular !== undefined) {
      resolved.specular = std_msgs.msg.ColorRGBA.Resolve(msg.specular)
    }
    else {
      resolved.specular = new std_msgs.msg.ColorRGBA()
    }

    if (msg.emissive !== undefined) {
      resolved.emissive = std_msgs.msg.ColorRGBA.Resolve(msg.emissive)
    }
    else {
      resolved.emissive = new std_msgs.msg.ColorRGBA()
    }

    return resolved;
    }
};

class SetLinkVisualPropertiesResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.success = null;
      this.status_message = null;
    }
    else {
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
    // Serializes a message object of type SetLinkVisualPropertiesResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    // Serialize message field [status_message]
    bufferOffset = _serializer.string(obj.status_message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetLinkVisualPropertiesResponse
    let len;
    let data = new SetLinkVisualPropertiesResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [status_message]
    data.status_message = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += object.status_message.length;
    return length + 5;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_ext_msgs/SetLinkVisualPropertiesResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '2ec6f3eff0161f4257b808b12bc830c2';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool success
    string status_message
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetLinkVisualPropertiesResponse(null);
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
  Request: SetLinkVisualPropertiesRequest,
  Response: SetLinkVisualPropertiesResponse,
  md5sum() { return 'defa425a32f63c1cdc8261ea0e650ab9'; },
  datatype() { return 'gazebo_ext_msgs/SetLinkVisualProperties'; }
};
