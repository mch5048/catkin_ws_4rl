; Auto-generated. Do not edit!


(cl:in-package berkeley_sawyer-srv)


;//! \htmlinclude get_kinectdata-request.msg.html

(cl:defclass <get_kinectdata-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass get_kinectdata-request (<get_kinectdata-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <get_kinectdata-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'get_kinectdata-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name berkeley_sawyer-srv:<get_kinectdata-request> is deprecated: use berkeley_sawyer-srv:get_kinectdata-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <get_kinectdata-request>) ostream)
  "Serializes a message object of type '<get_kinectdata-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <get_kinectdata-request>) istream)
  "Deserializes a message object of type '<get_kinectdata-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<get_kinectdata-request>)))
  "Returns string type for a service object of type '<get_kinectdata-request>"
  "berkeley_sawyer/get_kinectdataRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'get_kinectdata-request)))
  "Returns string type for a service object of type 'get_kinectdata-request"
  "berkeley_sawyer/get_kinectdataRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<get_kinectdata-request>)))
  "Returns md5sum for a message object of type '<get_kinectdata-request>"
  "b13d2865c5af2a64e6e30ab1b56e1dd5")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'get_kinectdata-request)))
  "Returns md5sum for a message object of type 'get_kinectdata-request"
  "b13d2865c5af2a64e6e30ab1b56e1dd5")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<get_kinectdata-request>)))
  "Returns full string definition for message of type '<get_kinectdata-request>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'get_kinectdata-request)))
  "Returns full string definition for message of type 'get_kinectdata-request"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <get_kinectdata-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <get_kinectdata-request>))
  "Converts a ROS message object to a list"
  (cl:list 'get_kinectdata-request
))
;//! \htmlinclude get_kinectdata-response.msg.html

(cl:defclass <get_kinectdata-response> (roslisp-msg-protocol:ros-message)
  ((image
    :reader image
    :initarg :image
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image)))
)

(cl:defclass get_kinectdata-response (<get_kinectdata-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <get_kinectdata-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'get_kinectdata-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name berkeley_sawyer-srv:<get_kinectdata-response> is deprecated: use berkeley_sawyer-srv:get_kinectdata-response instead.")))

(cl:ensure-generic-function 'image-val :lambda-list '(m))
(cl:defmethod image-val ((m <get_kinectdata-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader berkeley_sawyer-srv:image-val is deprecated.  Use berkeley_sawyer-srv:image instead.")
  (image m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <get_kinectdata-response>) ostream)
  "Serializes a message object of type '<get_kinectdata-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'image) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <get_kinectdata-response>) istream)
  "Deserializes a message object of type '<get_kinectdata-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'image) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<get_kinectdata-response>)))
  "Returns string type for a service object of type '<get_kinectdata-response>"
  "berkeley_sawyer/get_kinectdataResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'get_kinectdata-response)))
  "Returns string type for a service object of type 'get_kinectdata-response"
  "berkeley_sawyer/get_kinectdataResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<get_kinectdata-response>)))
  "Returns md5sum for a message object of type '<get_kinectdata-response>"
  "b13d2865c5af2a64e6e30ab1b56e1dd5")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'get_kinectdata-response)))
  "Returns md5sum for a message object of type 'get_kinectdata-response"
  "b13d2865c5af2a64e6e30ab1b56e1dd5")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<get_kinectdata-response>)))
  "Returns full string definition for message of type '<get_kinectdata-response>"
  (cl:format cl:nil "sensor_msgs/Image image~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of cameara~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'get_kinectdata-response)))
  "Returns full string definition for message of type 'get_kinectdata-response"
  (cl:format cl:nil "sensor_msgs/Image image~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of cameara~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <get_kinectdata-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'image))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <get_kinectdata-response>))
  "Converts a ROS message object to a list"
  (cl:list 'get_kinectdata-response
    (cl:cons ':image (image msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'get_kinectdata)))
  'get_kinectdata-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'get_kinectdata)))
  'get_kinectdata-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'get_kinectdata)))
  "Returns string type for a service object of type '<get_kinectdata>"
  "berkeley_sawyer/get_kinectdata")