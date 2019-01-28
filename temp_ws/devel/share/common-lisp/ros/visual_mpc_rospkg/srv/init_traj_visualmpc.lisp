; Auto-generated. Do not edit!


(cl:in-package visual_mpc_rospkg-srv)


;//! \htmlinclude init_traj_visualmpc-request.msg.html

(cl:defclass <init_traj_visualmpc-request> (roslisp-msg-protocol:ros-message)
  ((itr
    :reader itr
    :initarg :itr
    :type cl:integer
    :initform 0)
   (igrp
    :reader igrp
    :initarg :igrp
    :type cl:integer
    :initform 0)
   (goalmain
    :reader goalmain
    :initarg :goalmain
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image))
   (goalaux1
    :reader goalaux1
    :initarg :goalaux1
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image))
   (save_subdir
    :reader save_subdir
    :initarg :save_subdir
    :type cl:string
    :initform ""))
)

(cl:defclass init_traj_visualmpc-request (<init_traj_visualmpc-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <init_traj_visualmpc-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'init_traj_visualmpc-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name visual_mpc_rospkg-srv:<init_traj_visualmpc-request> is deprecated: use visual_mpc_rospkg-srv:init_traj_visualmpc-request instead.")))

(cl:ensure-generic-function 'itr-val :lambda-list '(m))
(cl:defmethod itr-val ((m <init_traj_visualmpc-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader visual_mpc_rospkg-srv:itr-val is deprecated.  Use visual_mpc_rospkg-srv:itr instead.")
  (itr m))

(cl:ensure-generic-function 'igrp-val :lambda-list '(m))
(cl:defmethod igrp-val ((m <init_traj_visualmpc-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader visual_mpc_rospkg-srv:igrp-val is deprecated.  Use visual_mpc_rospkg-srv:igrp instead.")
  (igrp m))

(cl:ensure-generic-function 'goalmain-val :lambda-list '(m))
(cl:defmethod goalmain-val ((m <init_traj_visualmpc-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader visual_mpc_rospkg-srv:goalmain-val is deprecated.  Use visual_mpc_rospkg-srv:goalmain instead.")
  (goalmain m))

(cl:ensure-generic-function 'goalaux1-val :lambda-list '(m))
(cl:defmethod goalaux1-val ((m <init_traj_visualmpc-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader visual_mpc_rospkg-srv:goalaux1-val is deprecated.  Use visual_mpc_rospkg-srv:goalaux1 instead.")
  (goalaux1 m))

(cl:ensure-generic-function 'save_subdir-val :lambda-list '(m))
(cl:defmethod save_subdir-val ((m <init_traj_visualmpc-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader visual_mpc_rospkg-srv:save_subdir-val is deprecated.  Use visual_mpc_rospkg-srv:save_subdir instead.")
  (save_subdir m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <init_traj_visualmpc-request>) ostream)
  "Serializes a message object of type '<init_traj_visualmpc-request>"
  (cl:let* ((signed (cl:slot-value msg 'itr)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'igrp)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 18446744073709551616) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) unsigned) ostream)
    )
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'goalmain) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'goalaux1) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'save_subdir))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'save_subdir))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <init_traj_visualmpc-request>) istream)
  "Deserializes a message object of type '<init_traj_visualmpc-request>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'itr) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'igrp) (cl:if (cl:< unsigned 9223372036854775808) unsigned (cl:- unsigned 18446744073709551616))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'goalmain) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'goalaux1) istream)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'save_subdir) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'save_subdir) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<init_traj_visualmpc-request>)))
  "Returns string type for a service object of type '<init_traj_visualmpc-request>"
  "visual_mpc_rospkg/init_traj_visualmpcRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'init_traj_visualmpc-request)))
  "Returns string type for a service object of type 'init_traj_visualmpc-request"
  "visual_mpc_rospkg/init_traj_visualmpcRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<init_traj_visualmpc-request>)))
  "Returns md5sum for a message object of type '<init_traj_visualmpc-request>"
  "212549b9c1a4ea535ff2ca3d14d779c5")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'init_traj_visualmpc-request)))
  "Returns md5sum for a message object of type 'init_traj_visualmpc-request"
  "212549b9c1a4ea535ff2ca3d14d779c5")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<init_traj_visualmpc-request>)))
  "Returns full string definition for message of type '<init_traj_visualmpc-request>"
  (cl:format cl:nil "int64 itr~%int64 igrp~%sensor_msgs/Image goalmain~%sensor_msgs/Image goalaux1~%string save_subdir~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of cameara~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'init_traj_visualmpc-request)))
  "Returns full string definition for message of type 'init_traj_visualmpc-request"
  (cl:format cl:nil "int64 itr~%int64 igrp~%sensor_msgs/Image goalmain~%sensor_msgs/Image goalaux1~%string save_subdir~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of cameara~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <init_traj_visualmpc-request>))
  (cl:+ 0
     8
     8
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'goalmain))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'goalaux1))
     4 (cl:length (cl:slot-value msg 'save_subdir))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <init_traj_visualmpc-request>))
  "Converts a ROS message object to a list"
  (cl:list 'init_traj_visualmpc-request
    (cl:cons ':itr (itr msg))
    (cl:cons ':igrp (igrp msg))
    (cl:cons ':goalmain (goalmain msg))
    (cl:cons ':goalaux1 (goalaux1 msg))
    (cl:cons ':save_subdir (save_subdir msg))
))
;//! \htmlinclude init_traj_visualmpc-response.msg.html

(cl:defclass <init_traj_visualmpc-response> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass init_traj_visualmpc-response (<init_traj_visualmpc-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <init_traj_visualmpc-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'init_traj_visualmpc-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name visual_mpc_rospkg-srv:<init_traj_visualmpc-response> is deprecated: use visual_mpc_rospkg-srv:init_traj_visualmpc-response instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <init_traj_visualmpc-response>) ostream)
  "Serializes a message object of type '<init_traj_visualmpc-response>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <init_traj_visualmpc-response>) istream)
  "Deserializes a message object of type '<init_traj_visualmpc-response>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<init_traj_visualmpc-response>)))
  "Returns string type for a service object of type '<init_traj_visualmpc-response>"
  "visual_mpc_rospkg/init_traj_visualmpcResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'init_traj_visualmpc-response)))
  "Returns string type for a service object of type 'init_traj_visualmpc-response"
  "visual_mpc_rospkg/init_traj_visualmpcResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<init_traj_visualmpc-response>)))
  "Returns md5sum for a message object of type '<init_traj_visualmpc-response>"
  "212549b9c1a4ea535ff2ca3d14d779c5")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'init_traj_visualmpc-response)))
  "Returns md5sum for a message object of type 'init_traj_visualmpc-response"
  "212549b9c1a4ea535ff2ca3d14d779c5")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<init_traj_visualmpc-response>)))
  "Returns full string definition for message of type '<init_traj_visualmpc-response>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'init_traj_visualmpc-response)))
  "Returns full string definition for message of type 'init_traj_visualmpc-response"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <init_traj_visualmpc-response>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <init_traj_visualmpc-response>))
  "Converts a ROS message object to a list"
  (cl:list 'init_traj_visualmpc-response
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'init_traj_visualmpc)))
  'init_traj_visualmpc-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'init_traj_visualmpc)))
  'init_traj_visualmpc-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'init_traj_visualmpc)))
  "Returns string type for a service object of type '<init_traj_visualmpc>"
  "visual_mpc_rospkg/init_traj_visualmpc")