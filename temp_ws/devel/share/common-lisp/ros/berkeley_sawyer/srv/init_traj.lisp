; Auto-generated. Do not edit!


(cl:in-package berkeley_sawyer-srv)


;//! \htmlinclude init_traj-request.msg.html

(cl:defclass <init_traj-request> (roslisp-msg-protocol:ros-message)
  ((itr
    :reader itr
    :initarg :itr
    :type cl:integer
    :initform 0)
   (igrp
    :reader igrp
    :initarg :igrp
    :type cl:integer
    :initform 0))
)

(cl:defclass init_traj-request (<init_traj-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <init_traj-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'init_traj-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name berkeley_sawyer-srv:<init_traj-request> is deprecated: use berkeley_sawyer-srv:init_traj-request instead.")))

(cl:ensure-generic-function 'itr-val :lambda-list '(m))
(cl:defmethod itr-val ((m <init_traj-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader berkeley_sawyer-srv:itr-val is deprecated.  Use berkeley_sawyer-srv:itr instead.")
  (itr m))

(cl:ensure-generic-function 'igrp-val :lambda-list '(m))
(cl:defmethod igrp-val ((m <init_traj-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader berkeley_sawyer-srv:igrp-val is deprecated.  Use berkeley_sawyer-srv:igrp instead.")
  (igrp m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <init_traj-request>) ostream)
  "Serializes a message object of type '<init_traj-request>"
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
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <init_traj-request>) istream)
  "Deserializes a message object of type '<init_traj-request>"
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
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<init_traj-request>)))
  "Returns string type for a service object of type '<init_traj-request>"
  "berkeley_sawyer/init_trajRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'init_traj-request)))
  "Returns string type for a service object of type 'init_traj-request"
  "berkeley_sawyer/init_trajRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<init_traj-request>)))
  "Returns md5sum for a message object of type '<init_traj-request>"
  "58b1b06ba616229bc38a06a0a0af5730")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'init_traj-request)))
  "Returns md5sum for a message object of type 'init_traj-request"
  "58b1b06ba616229bc38a06a0a0af5730")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<init_traj-request>)))
  "Returns full string definition for message of type '<init_traj-request>"
  (cl:format cl:nil "int64 itr~%int64 igrp~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'init_traj-request)))
  "Returns full string definition for message of type 'init_traj-request"
  (cl:format cl:nil "int64 itr~%int64 igrp~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <init_traj-request>))
  (cl:+ 0
     8
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <init_traj-request>))
  "Converts a ROS message object to a list"
  (cl:list 'init_traj-request
    (cl:cons ':itr (itr msg))
    (cl:cons ':igrp (igrp msg))
))
;//! \htmlinclude init_traj-response.msg.html

(cl:defclass <init_traj-response> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass init_traj-response (<init_traj-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <init_traj-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'init_traj-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name berkeley_sawyer-srv:<init_traj-response> is deprecated: use berkeley_sawyer-srv:init_traj-response instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <init_traj-response>) ostream)
  "Serializes a message object of type '<init_traj-response>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <init_traj-response>) istream)
  "Deserializes a message object of type '<init_traj-response>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<init_traj-response>)))
  "Returns string type for a service object of type '<init_traj-response>"
  "berkeley_sawyer/init_trajResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'init_traj-response)))
  "Returns string type for a service object of type 'init_traj-response"
  "berkeley_sawyer/init_trajResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<init_traj-response>)))
  "Returns md5sum for a message object of type '<init_traj-response>"
  "58b1b06ba616229bc38a06a0a0af5730")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'init_traj-response)))
  "Returns md5sum for a message object of type 'init_traj-response"
  "58b1b06ba616229bc38a06a0a0af5730")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<init_traj-response>)))
  "Returns full string definition for message of type '<init_traj-response>"
  (cl:format cl:nil "~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'init_traj-response)))
  "Returns full string definition for message of type 'init_traj-response"
  (cl:format cl:nil "~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <init_traj-response>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <init_traj-response>))
  "Converts a ROS message object to a list"
  (cl:list 'init_traj-response
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'init_traj)))
  'init_traj-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'init_traj)))
  'init_traj-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'init_traj)))
  "Returns string type for a service object of type '<init_traj>"
  "berkeley_sawyer/init_traj")