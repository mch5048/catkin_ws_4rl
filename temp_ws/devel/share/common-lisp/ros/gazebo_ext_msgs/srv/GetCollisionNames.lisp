; Auto-generated. Do not edit!


(cl:in-package gazebo_ext_msgs-srv)


;//! \htmlinclude GetCollisionNames-request.msg.html

(cl:defclass <GetCollisionNames-request> (roslisp-msg-protocol:ros-message)
  ((link_names
    :reader link_names
    :initarg :link_names
    :type (cl:vector cl:string)
   :initform (cl:make-array 0 :element-type 'cl:string :initial-element "")))
)

(cl:defclass GetCollisionNames-request (<GetCollisionNames-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetCollisionNames-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetCollisionNames-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_ext_msgs-srv:<GetCollisionNames-request> is deprecated: use gazebo_ext_msgs-srv:GetCollisionNames-request instead.")))

(cl:ensure-generic-function 'link_names-val :lambda-list '(m))
(cl:defmethod link_names-val ((m <GetCollisionNames-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:link_names-val is deprecated.  Use gazebo_ext_msgs-srv:link_names instead.")
  (link_names m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetCollisionNames-request>) ostream)
  "Serializes a message object of type '<GetCollisionNames-request>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'link_names))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((__ros_str_len (cl:length ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) ele))
   (cl:slot-value msg 'link_names))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetCollisionNames-request>) istream)
  "Deserializes a message object of type '<GetCollisionNames-request>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'link_names) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'link_names)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:aref vals i) __ros_str_idx) (cl:code-char (cl:read-byte istream))))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetCollisionNames-request>)))
  "Returns string type for a service object of type '<GetCollisionNames-request>"
  "gazebo_ext_msgs/GetCollisionNamesRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetCollisionNames-request)))
  "Returns string type for a service object of type 'GetCollisionNames-request"
  "gazebo_ext_msgs/GetCollisionNamesRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetCollisionNames-request>)))
  "Returns md5sum for a message object of type '<GetCollisionNames-request>"
  "440bd39ad26bc774c2ef6bcfe06d56ea")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetCollisionNames-request)))
  "Returns md5sum for a message object of type 'GetCollisionNames-request"
  "440bd39ad26bc774c2ef6bcfe06d56ea")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetCollisionNames-request>)))
  "Returns full string definition for message of type '<GetCollisionNames-request>"
  (cl:format cl:nil "string[] link_names~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetCollisionNames-request)))
  "Returns full string definition for message of type 'GetCollisionNames-request"
  (cl:format cl:nil "string[] link_names~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetCollisionNames-request>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'link_names) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4 (cl:length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetCollisionNames-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetCollisionNames-request
    (cl:cons ':link_names (link_names msg))
))
;//! \htmlinclude GetCollisionNames-response.msg.html

(cl:defclass <GetCollisionNames-response> (roslisp-msg-protocol:ros-message)
  ((link_collision_names
    :reader link_collision_names
    :initarg :link_collision_names
    :type (cl:vector cl:string)
   :initform (cl:make-array 0 :element-type 'cl:string :initial-element ""))
   (success
    :reader success
    :initarg :success
    :type cl:boolean
    :initform cl:nil)
   (status_message
    :reader status_message
    :initarg :status_message
    :type cl:string
    :initform ""))
)

(cl:defclass GetCollisionNames-response (<GetCollisionNames-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetCollisionNames-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetCollisionNames-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_ext_msgs-srv:<GetCollisionNames-response> is deprecated: use gazebo_ext_msgs-srv:GetCollisionNames-response instead.")))

(cl:ensure-generic-function 'link_collision_names-val :lambda-list '(m))
(cl:defmethod link_collision_names-val ((m <GetCollisionNames-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:link_collision_names-val is deprecated.  Use gazebo_ext_msgs-srv:link_collision_names instead.")
  (link_collision_names m))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <GetCollisionNames-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:success-val is deprecated.  Use gazebo_ext_msgs-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'status_message-val :lambda-list '(m))
(cl:defmethod status_message-val ((m <GetCollisionNames-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:status_message-val is deprecated.  Use gazebo_ext_msgs-srv:status_message instead.")
  (status_message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetCollisionNames-response>) ostream)
  "Serializes a message object of type '<GetCollisionNames-response>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'link_collision_names))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((__ros_str_len (cl:length ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) ele))
   (cl:slot-value msg 'link_collision_names))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'status_message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'status_message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetCollisionNames-response>) istream)
  "Deserializes a message object of type '<GetCollisionNames-response>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'link_collision_names) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'link_collision_names)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:aref vals i) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:aref vals i) __ros_str_idx) (cl:code-char (cl:read-byte istream))))))))
    (cl:setf (cl:slot-value msg 'success) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'status_message) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'status_message) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetCollisionNames-response>)))
  "Returns string type for a service object of type '<GetCollisionNames-response>"
  "gazebo_ext_msgs/GetCollisionNamesResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetCollisionNames-response)))
  "Returns string type for a service object of type 'GetCollisionNames-response"
  "gazebo_ext_msgs/GetCollisionNamesResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetCollisionNames-response>)))
  "Returns md5sum for a message object of type '<GetCollisionNames-response>"
  "440bd39ad26bc774c2ef6bcfe06d56ea")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetCollisionNames-response)))
  "Returns md5sum for a message object of type 'GetCollisionNames-response"
  "440bd39ad26bc774c2ef6bcfe06d56ea")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetCollisionNames-response>)))
  "Returns full string definition for message of type '<GetCollisionNames-response>"
  (cl:format cl:nil "string[] link_collision_names~%bool success~%string status_message~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetCollisionNames-response)))
  "Returns full string definition for message of type 'GetCollisionNames-response"
  (cl:format cl:nil "string[] link_collision_names~%bool success~%string status_message~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetCollisionNames-response>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'link_collision_names) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4 (cl:length ele))))
     1
     4 (cl:length (cl:slot-value msg 'status_message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetCollisionNames-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetCollisionNames-response
    (cl:cons ':link_collision_names (link_collision_names msg))
    (cl:cons ':success (success msg))
    (cl:cons ':status_message (status_message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetCollisionNames)))
  'GetCollisionNames-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetCollisionNames)))
  'GetCollisionNames-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetCollisionNames)))
  "Returns string type for a service object of type '<GetCollisionNames>"
  "gazebo_ext_msgs/GetCollisionNames")