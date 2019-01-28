; Auto-generated. Do not edit!


(cl:in-package gazebo_ext_msgs-srv)


;//! \htmlinclude GetSurfaceParams-request.msg.html

(cl:defclass <GetSurfaceParams-request> (roslisp-msg-protocol:ros-message)
  ((link_collision_name
    :reader link_collision_name
    :initarg :link_collision_name
    :type cl:string
    :initform ""))
)

(cl:defclass GetSurfaceParams-request (<GetSurfaceParams-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetSurfaceParams-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetSurfaceParams-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_ext_msgs-srv:<GetSurfaceParams-request> is deprecated: use gazebo_ext_msgs-srv:GetSurfaceParams-request instead.")))

(cl:ensure-generic-function 'link_collision_name-val :lambda-list '(m))
(cl:defmethod link_collision_name-val ((m <GetSurfaceParams-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:link_collision_name-val is deprecated.  Use gazebo_ext_msgs-srv:link_collision_name instead.")
  (link_collision_name m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetSurfaceParams-request>) ostream)
  "Serializes a message object of type '<GetSurfaceParams-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'link_collision_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'link_collision_name))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetSurfaceParams-request>) istream)
  "Deserializes a message object of type '<GetSurfaceParams-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'link_collision_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'link_collision_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetSurfaceParams-request>)))
  "Returns string type for a service object of type '<GetSurfaceParams-request>"
  "gazebo_ext_msgs/GetSurfaceParamsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSurfaceParams-request)))
  "Returns string type for a service object of type 'GetSurfaceParams-request"
  "gazebo_ext_msgs/GetSurfaceParamsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetSurfaceParams-request>)))
  "Returns md5sum for a message object of type '<GetSurfaceParams-request>"
  "ea04093721300487c8d44a7ccf00cb51")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetSurfaceParams-request)))
  "Returns md5sum for a message object of type 'GetSurfaceParams-request"
  "ea04093721300487c8d44a7ccf00cb51")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetSurfaceParams-request>)))
  "Returns full string definition for message of type '<GetSurfaceParams-request>"
  (cl:format cl:nil "string link_collision_name~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetSurfaceParams-request)))
  "Returns full string definition for message of type 'GetSurfaceParams-request"
  (cl:format cl:nil "string link_collision_name~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetSurfaceParams-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'link_collision_name))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetSurfaceParams-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetSurfaceParams-request
    (cl:cons ':link_collision_name (link_collision_name msg))
))
;//! \htmlinclude GetSurfaceParams-response.msg.html

(cl:defclass <GetSurfaceParams-response> (roslisp-msg-protocol:ros-message)
  ((elastic_modulus
    :reader elastic_modulus
    :initarg :elastic_modulus
    :type cl:float
    :initform 0.0)
   (mu1
    :reader mu1
    :initarg :mu1
    :type cl:float
    :initform 0.0)
   (mu2
    :reader mu2
    :initarg :mu2
    :type cl:float
    :initform 0.0)
   (mu_torsion
    :reader mu_torsion
    :initarg :mu_torsion
    :type cl:float
    :initform 0.0)
   (patch_radius
    :reader patch_radius
    :initarg :patch_radius
    :type cl:float
    :initform 0.0)
   (poisson_ratio
    :reader poisson_ratio
    :initarg :poisson_ratio
    :type cl:float
    :initform 0.0)
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

(cl:defclass GetSurfaceParams-response (<GetSurfaceParams-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetSurfaceParams-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetSurfaceParams-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_ext_msgs-srv:<GetSurfaceParams-response> is deprecated: use gazebo_ext_msgs-srv:GetSurfaceParams-response instead.")))

(cl:ensure-generic-function 'elastic_modulus-val :lambda-list '(m))
(cl:defmethod elastic_modulus-val ((m <GetSurfaceParams-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:elastic_modulus-val is deprecated.  Use gazebo_ext_msgs-srv:elastic_modulus instead.")
  (elastic_modulus m))

(cl:ensure-generic-function 'mu1-val :lambda-list '(m))
(cl:defmethod mu1-val ((m <GetSurfaceParams-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:mu1-val is deprecated.  Use gazebo_ext_msgs-srv:mu1 instead.")
  (mu1 m))

(cl:ensure-generic-function 'mu2-val :lambda-list '(m))
(cl:defmethod mu2-val ((m <GetSurfaceParams-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:mu2-val is deprecated.  Use gazebo_ext_msgs-srv:mu2 instead.")
  (mu2 m))

(cl:ensure-generic-function 'mu_torsion-val :lambda-list '(m))
(cl:defmethod mu_torsion-val ((m <GetSurfaceParams-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:mu_torsion-val is deprecated.  Use gazebo_ext_msgs-srv:mu_torsion instead.")
  (mu_torsion m))

(cl:ensure-generic-function 'patch_radius-val :lambda-list '(m))
(cl:defmethod patch_radius-val ((m <GetSurfaceParams-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:patch_radius-val is deprecated.  Use gazebo_ext_msgs-srv:patch_radius instead.")
  (patch_radius m))

(cl:ensure-generic-function 'poisson_ratio-val :lambda-list '(m))
(cl:defmethod poisson_ratio-val ((m <GetSurfaceParams-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:poisson_ratio-val is deprecated.  Use gazebo_ext_msgs-srv:poisson_ratio instead.")
  (poisson_ratio m))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <GetSurfaceParams-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:success-val is deprecated.  Use gazebo_ext_msgs-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'status_message-val :lambda-list '(m))
(cl:defmethod status_message-val ((m <GetSurfaceParams-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:status_message-val is deprecated.  Use gazebo_ext_msgs-srv:status_message instead.")
  (status_message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetSurfaceParams-response>) ostream)
  "Serializes a message object of type '<GetSurfaceParams-response>"
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'elastic_modulus))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'mu1))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'mu2))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'mu_torsion))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'patch_radius))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'poisson_ratio))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'status_message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'status_message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetSurfaceParams-response>) istream)
  "Deserializes a message object of type '<GetSurfaceParams-response>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'elastic_modulus) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'mu1) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'mu2) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'mu_torsion) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'patch_radius) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'poisson_ratio) (roslisp-utils:decode-double-float-bits bits)))
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetSurfaceParams-response>)))
  "Returns string type for a service object of type '<GetSurfaceParams-response>"
  "gazebo_ext_msgs/GetSurfaceParamsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSurfaceParams-response)))
  "Returns string type for a service object of type 'GetSurfaceParams-response"
  "gazebo_ext_msgs/GetSurfaceParamsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetSurfaceParams-response>)))
  "Returns md5sum for a message object of type '<GetSurfaceParams-response>"
  "ea04093721300487c8d44a7ccf00cb51")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetSurfaceParams-response)))
  "Returns md5sum for a message object of type 'GetSurfaceParams-response"
  "ea04093721300487c8d44a7ccf00cb51")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetSurfaceParams-response>)))
  "Returns full string definition for message of type '<GetSurfaceParams-response>"
  (cl:format cl:nil "float64 elastic_modulus~%float64 mu1~%float64 mu2~%float64 mu_torsion~%float64 patch_radius~%float64 poisson_ratio~%bool success~%string status_message~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetSurfaceParams-response)))
  "Returns full string definition for message of type 'GetSurfaceParams-response"
  (cl:format cl:nil "float64 elastic_modulus~%float64 mu1~%float64 mu2~%float64 mu_torsion~%float64 patch_radius~%float64 poisson_ratio~%bool success~%string status_message~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetSurfaceParams-response>))
  (cl:+ 0
     8
     8
     8
     8
     8
     8
     1
     4 (cl:length (cl:slot-value msg 'status_message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetSurfaceParams-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetSurfaceParams-response
    (cl:cons ':elastic_modulus (elastic_modulus msg))
    (cl:cons ':mu1 (mu1 msg))
    (cl:cons ':mu2 (mu2 msg))
    (cl:cons ':mu_torsion (mu_torsion msg))
    (cl:cons ':patch_radius (patch_radius msg))
    (cl:cons ':poisson_ratio (poisson_ratio msg))
    (cl:cons ':success (success msg))
    (cl:cons ':status_message (status_message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetSurfaceParams)))
  'GetSurfaceParams-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetSurfaceParams)))
  'GetSurfaceParams-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSurfaceParams)))
  "Returns string type for a service object of type '<GetSurfaceParams>"
  "gazebo_ext_msgs/GetSurfaceParams")