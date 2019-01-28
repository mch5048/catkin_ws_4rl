; Auto-generated. Do not edit!


(cl:in-package rospy_message_converter-msg)


;//! \htmlinclude Uint8ArrayTestMessage.msg.html

(cl:defclass <Uint8ArrayTestMessage> (roslisp-msg-protocol:ros-message)
  ((data
    :reader data
    :initarg :data
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 0 :element-type 'cl:fixnum :initial-element 0)))
)

(cl:defclass Uint8ArrayTestMessage (<Uint8ArrayTestMessage>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Uint8ArrayTestMessage>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Uint8ArrayTestMessage)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rospy_message_converter-msg:<Uint8ArrayTestMessage> is deprecated: use rospy_message_converter-msg:Uint8ArrayTestMessage instead.")))

(cl:ensure-generic-function 'data-val :lambda-list '(m))
(cl:defmethod data-val ((m <Uint8ArrayTestMessage>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rospy_message_converter-msg:data-val is deprecated.  Use rospy_message_converter-msg:data instead.")
  (data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Uint8ArrayTestMessage>) ostream)
  "Serializes a message object of type '<Uint8ArrayTestMessage>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'data))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'data))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Uint8ArrayTestMessage>) istream)
  "Deserializes a message object of type '<Uint8ArrayTestMessage>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'data) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'data)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Uint8ArrayTestMessage>)))
  "Returns string type for a message object of type '<Uint8ArrayTestMessage>"
  "rospy_message_converter/Uint8ArrayTestMessage")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Uint8ArrayTestMessage)))
  "Returns string type for a message object of type 'Uint8ArrayTestMessage"
  "rospy_message_converter/Uint8ArrayTestMessage")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Uint8ArrayTestMessage>)))
  "Returns md5sum for a message object of type '<Uint8ArrayTestMessage>"
  "f43a8e1b362b75baa741461b46adc7e0")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Uint8ArrayTestMessage)))
  "Returns md5sum for a message object of type 'Uint8ArrayTestMessage"
  "f43a8e1b362b75baa741461b46adc7e0")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Uint8ArrayTestMessage>)))
  "Returns full string definition for message of type '<Uint8ArrayTestMessage>"
  (cl:format cl:nil "# Size-agnostic uint8 array for testing purposes~%uint8[] data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Uint8ArrayTestMessage)))
  "Returns full string definition for message of type 'Uint8ArrayTestMessage"
  (cl:format cl:nil "# Size-agnostic uint8 array for testing purposes~%uint8[] data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Uint8ArrayTestMessage>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'data) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Uint8ArrayTestMessage>))
  "Converts a ROS message object to a list"
  (cl:list 'Uint8ArrayTestMessage
    (cl:cons ':data (data msg))
))
