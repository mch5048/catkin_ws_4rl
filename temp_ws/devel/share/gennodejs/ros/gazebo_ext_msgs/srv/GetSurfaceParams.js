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

class GetSurfaceParamsRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.link_collision_name = null;
    }
    else {
      if (initObj.hasOwnProperty('link_collision_name')) {
        this.link_collision_name = initObj.link_collision_name
      }
      else {
        this.link_collision_name = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetSurfaceParamsRequest
    // Serialize message field [link_collision_name]
    bufferOffset = _serializer.string(obj.link_collision_name, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetSurfaceParamsRequest
    let len;
    let data = new GetSurfaceParamsRequest(null);
    // Deserialize message field [link_collision_name]
    data.link_collision_name = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += object.link_collision_name.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_ext_msgs/GetSurfaceParamsRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'fc16c45e57c83223544149fc78b39abb';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string link_collision_name
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetSurfaceParamsRequest(null);
    if (msg.link_collision_name !== undefined) {
      resolved.link_collision_name = msg.link_collision_name;
    }
    else {
      resolved.link_collision_name = ''
    }

    return resolved;
    }
};

class GetSurfaceParamsResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.elastic_modulus = null;
      this.mu1 = null;
      this.mu2 = null;
      this.mu_torsion = null;
      this.patch_radius = null;
      this.poisson_ratio = null;
      this.success = null;
      this.status_message = null;
    }
    else {
      if (initObj.hasOwnProperty('elastic_modulus')) {
        this.elastic_modulus = initObj.elastic_modulus
      }
      else {
        this.elastic_modulus = 0.0;
      }
      if (initObj.hasOwnProperty('mu1')) {
        this.mu1 = initObj.mu1
      }
      else {
        this.mu1 = 0.0;
      }
      if (initObj.hasOwnProperty('mu2')) {
        this.mu2 = initObj.mu2
      }
      else {
        this.mu2 = 0.0;
      }
      if (initObj.hasOwnProperty('mu_torsion')) {
        this.mu_torsion = initObj.mu_torsion
      }
      else {
        this.mu_torsion = 0.0;
      }
      if (initObj.hasOwnProperty('patch_radius')) {
        this.patch_radius = initObj.patch_radius
      }
      else {
        this.patch_radius = 0.0;
      }
      if (initObj.hasOwnProperty('poisson_ratio')) {
        this.poisson_ratio = initObj.poisson_ratio
      }
      else {
        this.poisson_ratio = 0.0;
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
    // Serializes a message object of type GetSurfaceParamsResponse
    // Serialize message field [elastic_modulus]
    bufferOffset = _serializer.float64(obj.elastic_modulus, buffer, bufferOffset);
    // Serialize message field [mu1]
    bufferOffset = _serializer.float64(obj.mu1, buffer, bufferOffset);
    // Serialize message field [mu2]
    bufferOffset = _serializer.float64(obj.mu2, buffer, bufferOffset);
    // Serialize message field [mu_torsion]
    bufferOffset = _serializer.float64(obj.mu_torsion, buffer, bufferOffset);
    // Serialize message field [patch_radius]
    bufferOffset = _serializer.float64(obj.patch_radius, buffer, bufferOffset);
    // Serialize message field [poisson_ratio]
    bufferOffset = _serializer.float64(obj.poisson_ratio, buffer, bufferOffset);
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    // Serialize message field [status_message]
    bufferOffset = _serializer.string(obj.status_message, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetSurfaceParamsResponse
    let len;
    let data = new GetSurfaceParamsResponse(null);
    // Deserialize message field [elastic_modulus]
    data.elastic_modulus = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [mu1]
    data.mu1 = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [mu2]
    data.mu2 = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [mu_torsion]
    data.mu_torsion = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [patch_radius]
    data.patch_radius = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [poisson_ratio]
    data.poisson_ratio = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [status_message]
    data.status_message = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += object.status_message.length;
    return length + 53;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_ext_msgs/GetSurfaceParamsResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '034c689e7f76f63e9bd3514125ee3535';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float64 elastic_modulus
    float64 mu1
    float64 mu2
    float64 mu_torsion
    float64 patch_radius
    float64 poisson_ratio
    bool success
    string status_message
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetSurfaceParamsResponse(null);
    if (msg.elastic_modulus !== undefined) {
      resolved.elastic_modulus = msg.elastic_modulus;
    }
    else {
      resolved.elastic_modulus = 0.0
    }

    if (msg.mu1 !== undefined) {
      resolved.mu1 = msg.mu1;
    }
    else {
      resolved.mu1 = 0.0
    }

    if (msg.mu2 !== undefined) {
      resolved.mu2 = msg.mu2;
    }
    else {
      resolved.mu2 = 0.0
    }

    if (msg.mu_torsion !== undefined) {
      resolved.mu_torsion = msg.mu_torsion;
    }
    else {
      resolved.mu_torsion = 0.0
    }

    if (msg.patch_radius !== undefined) {
      resolved.patch_radius = msg.patch_radius;
    }
    else {
      resolved.patch_radius = 0.0
    }

    if (msg.poisson_ratio !== undefined) {
      resolved.poisson_ratio = msg.poisson_ratio;
    }
    else {
      resolved.poisson_ratio = 0.0
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
  Request: GetSurfaceParamsRequest,
  Response: GetSurfaceParamsResponse,
  md5sum() { return 'ea04093721300487c8d44a7ccf00cb51'; },
  datatype() { return 'gazebo_ext_msgs/GetSurfaceParams'; }
};
