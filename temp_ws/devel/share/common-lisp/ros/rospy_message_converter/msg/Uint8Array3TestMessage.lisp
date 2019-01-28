; Auto-generated. Do not edit!


(cl:in-package rospy_message_converter-msg)


;//! \htmlinclude Uint8Array3TestMessage.msg.html

(cl:defclass <Uint8Array3TestMessage> (roslisp-msg-protocol:ros-message)
  ((data
    :reader data
    :initarg :data
    :type (cl:vector cl:fixnum)
   :initform (cl:make-array 3 :element-type 'cl:fixnum :initial-element 0)))
)

(cl:defclass Uint8Array3TestMessage (<Uint8Array3TestMessage>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Uint8Array3TestMessage>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Uint8Array3TestMessage)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rospy_message_converter-msg:<Uint8Array3TestMessage> is deprecated: use rospy_message_converter-msg:Uint8Array3TestMessage instead.")))

(cl:ensure-generic-function 'data-val :lambda-list '(m))
(cl:defmethod data-val ((m <Uint8Array3TestMessage>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rospy_message_converter-msg:data-val is deprecated.  Use rospy_message_converter-msg:data instead.")
  (data m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Uint8Array3TestMessage>) ostream)
  "Serializes a message object of type '<Uint8Array3TestMessage>"
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream))
   (cl:slot-value msg 'data))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Uint8Array3TestMessage>) istream)
  "Deserializes a message object of type '<Uint8Array3TestMessage>"
  (cl:setf (cl:slot-value msg 'data) (cl:make-array 3))
  (cl:let ((vals (cl:slot-value msg 'data)))
    (cl:dotimes (i 3)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Uint8Array3TestMessage>)))
  "Returns string type for a message object of type '<Uint8Array3TestMessage>"
  "rospy_message_converter/Uint8Array3TestMessage")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Uint8Array3TestMessage)))
  "Returns string type for a message object of type 'Uint8Array3TestMessage"
  "rospy_message_converter/Uint8Array3TestMessage")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Uint8Array3TestMessage>)))
  "Returns md5sum for a message object of type '<Uint8Array3TestMessage>"
  "0aff379edf54a041a83af067122e7bc6")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Uint8Array3TestMessage)))
  "Returns md5sum for a message object of type 'Uint8Array3TestMessage"
  "0aff379edf54a041a83af067122e7bc6")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Uint8Array3TestMessage>)))
  "Returns full string definition for message of type '<Uint8Array3TestMessage>"
  (cl:format cl:nil "# Fixed size uint8 array for testing purposes~%uint8[3] data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Uint8Array3TestMessage)))
  "Returns full string definition for message of type 'Uint8Array3TestMessage"
  (cl:format cl:nil "# Fixed size uint8 array for testing purposes~%uint8[3] data~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Uint8Array3TestMessage>))
  (cl:+ 0
     0 (cl:reduce #'cl:+ (cl:slot-value msg 'data) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 1)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Uint8Array3TestMessage>))
  "Converts a ROS message object to a list"
  (cl:list 'Uint8Array3TestMessage
    (cl:cons ':data (data msg))
))
