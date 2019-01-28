; Auto-generated. Do not edit!


(cl:in-package sawyer_sim_examples-msg)


;//! \htmlinclude GoalObs.msg.html

(cl:defclass <GoalObs> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (goal_cart_pos
    :reader goal_cart_pos
    :initarg :goal_cart_pos
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass GoalObs (<GoalObs>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GoalObs>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GoalObs)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name sawyer_sim_examples-msg:<GoalObs> is deprecated: use sawyer_sim_examples-msg:GoalObs instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <GoalObs>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sawyer_sim_examples-msg:header-val is deprecated.  Use sawyer_sim_examples-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'goal_cart_pos-val :lambda-list '(m))
(cl:defmethod goal_cart_pos-val ((m <GoalObs>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader sawyer_sim_examples-msg:goal_cart_pos-val is deprecated.  Use sawyer_sim_examples-msg:goal_cart_pos instead.")
  (goal_cart_pos m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GoalObs>) ostream)
  "Serializes a message object of type '<GoalObs>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'goal_cart_pos))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-double-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream)))
   (cl:slot-value msg 'goal_cart_pos))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GoalObs>) istream)
  "Deserializes a message object of type '<GoalObs>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'goal_cart_pos) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'goal_cart_pos)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-double-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GoalObs>)))
  "Returns string type for a message object of type '<GoalObs>"
  "sawyer_sim_examples/GoalObs")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GoalObs)))
  "Returns string type for a message object of type 'GoalObs"
  "sawyer_sim_examples/GoalObs")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GoalObs>)))
  "Returns md5sum for a message object of type '<GoalObs>"
  "2a7aed1f0b63cf2c59e8c977d20aef5b")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GoalObs)))
  "Returns md5sum for a message object of type 'GoalObs"
  "2a7aed1f0b63cf2c59e8c977d20aef5b")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GoalObs>)))
  "Returns full string definition for message of type '<GoalObs>"
  (cl:format cl:nil "##the name of the output~%Header header~%~%float64[] goal_cart_pos~%~%##the value to set output ~%~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GoalObs)))
  "Returns full string definition for message of type 'GoalObs"
  (cl:format cl:nil "##the name of the output~%Header header~%~%float64[] goal_cart_pos~%~%##the value to set output ~%~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GoalObs>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'goal_cart_pos) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 8)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GoalObs>))
  "Converts a ROS message object to a list"
  (cl:list 'GoalObs
    (cl:cons ':header (header msg))
    (cl:cons ':goal_cart_pos (goal_cart_pos msg))
))
