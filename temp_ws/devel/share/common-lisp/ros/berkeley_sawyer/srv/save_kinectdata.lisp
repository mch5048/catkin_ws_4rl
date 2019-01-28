; Auto-generated. Do not edit!


(cl:in-package berkeley_sawyer-srv)


;//! \htmlinclude save_kinectdata-request.msg.html

(cl:defclass <save_kinectdata-request> (roslisp-msg-protocol:ros-message)
  ((itr
    :reader itr
    :initarg :itr
    :type cl:integer
    :initform 0))
)

(cl:defclass save_kinectdata-request (<save_kinectdata-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <save_kinectdata-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'save_kinectdata-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name berkeley_sawyer-srv:<save_kinectdata-request> is deprecated: use berkeley_sawyer-srv:save_kinectdata-request instead.")))

(cl:ensure-generic-function 'itr-val :lambda-list '(m))
(cl:defmethod itr-val ((m <save_kinectdata-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader berkeley_sawyer-srv:itr-val is deprecated.  Use berkeley_sawyer-srv:itr instead.")
  (itr m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <save_kinectdata-request>) ostream)
  "Serializes a message object of type '<save_kinectdata-request>"
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
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <save_kinectdata-request>) istream)
  "Deserializes a message object of type '<save_kinectdata-request>"
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
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<save_kinectdata-request>)))
  "Returns string type for a service object of type '<save_kinectdata-request>"
  "berkeley_sawyer/save_kinectdataRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'save_kinectdata-request)))
  "Returns string type for a service object of type 'save_kinectdata-request"
  "berkeley_sawyer/save_kinectdataRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<save_kinectdata-request>)))
  "Returns md5sum for a message object of type '<save_kinectdata-request>"
  "36618b1eaec8d2483d42a27cb2744012")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'save_kinectdata-request)))
  "Returns md5sum for a message object of type 'save_kinectdata-request"
  "36618b1eaec8d2483d42a27cb2744012")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<save_kinectdata-request>)))
  "Returns full string definition for message of type '<save_kinectdata-request>"
  (cl:format cl:nil "int64 itr~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'save_kinectdata-request)))
  "Returns full string definition for message of type 'save_kinectdata-request"
  (cl:format cl:nil "int64 itr~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <save_kinectdata-request>))
  (cl:+ 0
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <save_kinectdata-request>))
  "Converts a ROS message object to a list"
  (cl:list 'save_kinectdata-request
    (cl:cons ':itr (itr msg))
))
;//! \htmlinclude save_kinectdata-response.msg.html

(cl:defclass <save_kinectdata-response> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass save_kinectdata-response (<save_kinectdata-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <save_kinectdata-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'save_kinectdata-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name berkeley_sawyer-srv:<save_kinectdata-response> is deprecated: use berkeley_sawyer-srv:save_kinectdata-response instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <save_kinectdata-response>) ostream)
  "Serializes a message object of type '<save_kinectdata-response>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <save_kinectdata-response>) istream)
  "Deserializes a message object of type '<save_kinectdata-response>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<save_kinectdata-response>)))
  "Returns string type for a service object of type '<save_kinectdata-response>"
  "berkeley_sawyer/save_kinectdataResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'save_kinectdata-response)))
  "Returns string type for a service object of type 'save_kinectdata-response"
  "berkeley_sawyer/save_kinectdataResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<save_kinectdata-response>)))
  "Returns md5sum for a message object of type '<save_kinectdata-response>"
  "36618b1eaec8d2483d42a27cb2744012")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'save_kinectdata-response)))
  "Returns md5sum for a message object of type 'save_kinectdata-response"
  "36618b1eaec8d2483d42a27cb2744012")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<save_kinectdata-response>)))
  "Returns full string definition for message of type '<save_kinectdata-response>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'save_kinectdata-response)))
  "Returns full string definition for message of type 'save_kinectdata-response"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <save_kinectdata-response>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <save_kinectdata-response>))
  "Converts a ROS message object to a list"
  (cl:list 'save_kinectdata-response
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'save_kinectdata)))
  'save_kinectdata-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'save_kinectdata)))
  'save_kinectdata-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'save_kinectdata)))
  "Returns string type for a service object of type '<save_kinectdata>"
  "berkeley_sawyer/save_kinectdata")