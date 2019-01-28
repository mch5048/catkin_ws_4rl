; Auto-generated. Do not edit!


(cl:in-package object_detection_yolov2-msg)


;//! \htmlinclude DetectionArray.msg.html

(cl:defclass <DetectionArray> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (size
    :reader size
    :initarg :size
    :type cl:integer
    :initform 0)
   (data
    :reader data
    :initarg :data
    :type (cl:vector object_detection_yolov2-msg:Detection)
   :initform (cl:make-array 0 :element-type 'object_detection_yolov2-msg:Detection :initial-element (cl:make-instance 'object_detection_yolov2-msg:Detection))))
)

(cl:defclass DetectionArray (<DetectionArray>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DetectionArray>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DetectionArray)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name object_detection_yolov2-msg:<DetectionArray> is deprecated: use object_detection_yolov2-msg:DetectionArray instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <DetectionArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_detection_yolov2-msg:header-val is deprecated.  Use object_detection_yolov2-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'size-val :lambda-list '(m))
(cl:defmethod size-val ((m <DetectionArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_detection_yolov2-msg:size-val is deprecated.  Use object_detection_yolov2-msg:size instead.")
  (size m))

(cl:ensure-generic-function 'data-val :lambda-list '(m))
(cl:defmethod data-val ((m <DetectionArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_detection_yolov2-msg:data-val is deprecated.  Use object_detection_yolov2-msg:data instead.")
  (data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DetectionArray>) ostream)
  "Serializes a message object of type '<DetectionArray>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'size)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'size)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'size)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'size)) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'data))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'data))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DetectionArray>) istream)
  "Deserializes a message object of type '<DetectionArray>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'size)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'size)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'size)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'size)) (cl:read-byte istream))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'data) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'data)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'object_detection_yolov2-msg:Detection))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DetectionArray>)))
  "Returns string type for a message object of type '<DetectionArray>"
  "object_detection_yolov2/DetectionArray")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DetectionArray)))
  "Returns string type for a message object of type 'DetectionArray"
  "object_detection_yolov2/DetectionArray")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DetectionArray>)))
  "Returns md5sum for a message object of type '<DetectionArray>"
  "f6f31e50896912f32e444660e6057cbd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DetectionArray)))
  "Returns md5sum for a message object of type 'DetectionArray"
  "f6f31e50896912f32e444660e6057cbd")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DetectionArray>)))
  "Returns full string definition for message of type '<DetectionArray>"
  (cl:format cl:nil "Header header~%~%# The size of the array~%uint32 size~%# The array containing all the detections~%Detection[] data~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%================================================================================~%MSG: object_detection_yolov2/Detection~%Header header~%~%string object_class~%float32 p~%~%uint16 x~%uint16 y~%~%float32 cam_x~%float32 cam_y~%float32 cam_z~%~%uint16 width~%uint16 height~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DetectionArray)))
  "Returns full string definition for message of type 'DetectionArray"
  (cl:format cl:nil "Header header~%~%# The size of the array~%uint32 size~%# The array containing all the detections~%Detection[] data~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%================================================================================~%MSG: object_detection_yolov2/Detection~%Header header~%~%string object_class~%float32 p~%~%uint16 x~%uint16 y~%~%float32 cam_x~%float32 cam_y~%float32 cam_z~%~%uint16 width~%uint16 height~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DetectionArray>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'data) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DetectionArray>))
  "Converts a ROS message object to a list"
  (cl:list 'DetectionArray
    (cl:cons ':header (header msg))
    (cl:cons ':size (size msg))
    (cl:cons ':data (data msg))
))
