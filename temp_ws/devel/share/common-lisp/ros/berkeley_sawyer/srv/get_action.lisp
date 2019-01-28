; Auto-generated. Do not edit!


(cl:in-package berkeley_sawyer-srv)


;//! \htmlinclude get_action-request.msg.html

(cl:defclass <get_action-request> (roslisp-msg-protocol:ros-message)
  ((main
    :reader main
    :initarg :main
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image))
   (aux1
    :reader aux1
    :initarg :aux1
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image))
   (state
    :reader state
    :initarg :state
    :type (cl:vector cl:float)
   :initform (cl:make-array 3 :element-type 'cl:float :initial-element 0.0))
   (desig_pos_aux1
    :reader desig_pos_aux1
    :initarg :desig_pos_aux1
    :type (cl:vector cl:integer)
   :initform (cl:make-array 4 :element-type 'cl:integer :initial-element 0))
   (goal_pos_aux1
    :reader goal_pos_aux1
    :initarg :goal_pos_aux1
    :type (cl:vector cl:integer)
   :initform (cl:make-array 4 :element-type 'cl:integer :initial-element 0)))
)

(cl:defclass get_action-request (<get_action-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <get_action-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'get_action-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name berkeley_sawyer-srv:<get_action-request> is deprecated: use berkeley_sawyer-srv:get_action-request instead.")))

(cl:ensure-generic-function 'main-val :lambda-list '(m))
(cl:defmethod main-val ((m <get_action-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader berkeley_sawyer-srv:main-val is deprecated.  Use berkeley_sawyer-srv:main instead.")
  (main m))

(cl:ensure-generic-function 'aux1-val :lambda-list '(m))
(cl:defmethod aux1-val ((m <get_action-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader berkeley_sawyer-srv:aux1-val is deprecated.  Use berkeley_sawyer-srv:aux1 instead.")
  (aux1 m))

(cl:ensure-generic-function 'state-val :lambda-list '(m))
(cl:defmethod state-val ((m <get_action-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader berkeley_sawyer-srv:state-val is deprecated.  Use berkeley_sawyer-srv:state instead.")
  (state m))

(cl:ensure-generic-function 'desig_pos_aux1-val :lambda-list '(m))
(cl:defmethod desig_pos_aux1-val ((m <get_action-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader berkeley_sawyer-srv:desig_pos_aux1-val is deprecated.  Use berkeley_sawyer-srv:desig_pos_aux1 instead.")
  (desig_pos_aux1 m))

(cl:ensure-generic-function 'goal_pos_aux1-val :lambda-list '(m))
(cl:defmethod goal_pos_aux1-val ((m <get_action-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader berkeley_sawyer-srv:goal_pos_aux1-val is deprecated.  Use berkeley_sawyer-srv:goal_pos_aux1 instead.")
  (goal_pos_aux1 m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <get_action-request>) ostream)
  "Serializes a message object of type '<get_action-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'main) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'aux1) ostream)
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'state))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) unsigned) ostream)
    ))
   (cl:slot-value msg 'desig_pos_aux1))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let* ((signed ele) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) unsigned) ostream)
    ))
   (cl:slot-value msg 'goal_pos_aux1))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <get_action-request>) istream)
  "Deserializes a message object of type '<get_action-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'main) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'aux1) istream)
  (cl:setf (cl:slot-value msg 'state) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'state)))
    (cl:dotimes (i 3)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  (cl:setf (cl:slot-value msg 'desig_pos_aux1) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'desig_pos_aux1)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))))
  (cl:setf (cl:slot-value msg 'goal_pos_aux1) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'goal_pos_aux1)))
    (cl:dotimes (i 4)
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<get_action-request>)))
  "Returns string type for a service object of type '<get_action-request>"
  "berkeley_sawyer/get_actionRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'get_action-request)))
  "Returns string type for a service object of type 'get_action-request"
  "berkeley_sawyer/get_actionRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<get_action-request>)))
  "Returns md5sum for a message object of type '<get_action-request>"
  "ed42212579cedad52c84913bafdfccf2")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'get_action-request)))
  "Returns md5sum for a message object of type 'get_action-request"
  "ed42212579cedad52c84913bafdfccf2")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<get_action-request>)))
  "Returns full string definition for message of type '<get_action-request>"
  (cl:format cl:nil "sensor_msgs/Image main~%sensor_msgs/Image aux1~%float32[3] state~%int64[4] desig_pos_aux1~%int64[4] goal_pos_aux1~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of cameara~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'get_action-request)))
  "Returns full string definition for message of type 'get_action-request"
  (cl:format cl:nil "sensor_msgs/Image main~%sensor_msgs/Image aux1~%float32[3] state~%int64[4] desig_pos_aux1~%int64[4] goal_pos_aux1~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of cameara~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <get_action-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'main))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'aux1))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'state) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'desig_pos_aux1) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'goal_pos_aux1) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <get_action-request>))
  "Converts a ROS message object to a list"
  (cl:list 'get_action-request
    (cl:cons ':main (main msg))
    (cl:cons ':aux1 (aux1 msg))
    (cl:cons ':state (state msg))
    (cl:cons ':desig_pos_aux1 (desig_pos_aux1 msg))
    (cl:cons ':goal_pos_aux1 (goal_pos_aux1 msg))
))
;//! \htmlinclude get_action-response.msg.html

(cl:defclass <get_action-response> (roslisp-msg-protocol:ros-message)
  ((action
    :reader action
    :initarg :action
    :type (cl:vector cl:float)
   :initform (cl:make-array 4 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass get_action-response (<get_action-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <get_action-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'get_action-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name berkeley_sawyer-srv:<get_action-response> is deprecated: use berkeley_sawyer-srv:get_action-response instead.")))

(cl:ensure-generic-function 'action-val :lambda-list '(m))
(cl:defmethod action-val ((m <get_action-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader berkeley_sawyer-srv:action-val is deprecated.  Use berkeley_sawyer-srv:action instead.")
  (action m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <get_action-response>) ostream)
  "Serializes a message object of type '<get_action-response>"
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'action))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <get_action-response>) istream)
  "Deserializes a message object of type '<get_action-response>"
  (cl:setf (cl:slot-value msg 'action) (cl:make-array 4))
  (cl:let ((vals (cl:slot-value msg 'action)))
    (cl:dotimes (i 4)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<get_action-response>)))
  "Returns string type for a service object of type '<get_action-response>"
  "berkeley_sawyer/get_actionResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'get_action-response)))
  "Returns string type for a service object of type 'get_action-response"
  "berkeley_sawyer/get_actionResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<get_action-response>)))
  "Returns md5sum for a message object of type '<get_action-response>"
  "ed42212579cedad52c84913bafdfccf2")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'get_action-response)))
  "Returns md5sum for a message object of type 'get_action-response"
  "ed42212579cedad52c84913bafdfccf2")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<get_action-response>)))
  "Returns full string definition for message of type '<get_action-response>"
  (cl:format cl:nil "float32[4] action~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'get_action-response)))
  "Returns full string definition for message of type 'get_action-response"
  (cl:format cl:nil "float32[4] action~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <get_action-response>))
  (cl:+ 0
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'action) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <get_action-response>))
  "Converts a ROS message object to a list"
  (cl:list 'get_action-response
    (cl:cons ':action (action msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'get_action)))
  'get_action-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'get_action)))
  'get_action-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'get_action)))
  "Returns string type for a service object of type '<get_action>"
  "berkeley_sawyer/get_action")