; Auto-generated. Do not edit!


(cl:in-package gazebo_ext_msgs-srv)


;//! \htmlinclude SetSurfaceParams-request.msg.html

(cl:defclass <SetSurfaceParams-request> (roslisp-msg-protocol:ros-message)
  ((link_collision_name
    :reader link_collision_name
    :initarg :link_collision_name
    :type cl:string
    :initform "")
   (elastic_modulus
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
    :initform 0.0))
)

(cl:defclass SetSurfaceParams-request (<SetSurfaceParams-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetSurfaceParams-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetSurfaceParams-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_ext_msgs-srv:<SetSurfaceParams-request> is deprecated: use gazebo_ext_msgs-srv:SetSurfaceParams-request instead.")))

(cl:ensure-generic-function 'link_collision_name-val :lambda-list '(m))
(cl:defmethod link_collision_name-val ((m <SetSurfaceParams-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:link_collision_name-val is deprecated.  Use gazebo_ext_msgs-srv:link_collision_name instead.")
  (link_collision_name m))

(cl:ensure-generic-function 'elastic_modulus-val :lambda-list '(m))
(cl:defmethod elastic_modulus-val ((m <SetSurfaceParams-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:elastic_modulus-val is deprecated.  Use gazebo_ext_msgs-srv:elastic_modulus instead.")
  (elastic_modulus m))

(cl:ensure-generic-function 'mu1-val :lambda-list '(m))
(cl:defmethod mu1-val ((m <SetSurfaceParams-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:mu1-val is deprecated.  Use gazebo_ext_msgs-srv:mu1 instead.")
  (mu1 m))

(cl:ensure-generic-function 'mu2-val :lambda-list '(m))
(cl:defmethod mu2-val ((m <SetSurfaceParams-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:mu2-val is deprecated.  Use gazebo_ext_msgs-srv:mu2 instead.")
  (mu2 m))

(cl:ensure-generic-function 'mu_torsion-val :lambda-list '(m))
(cl:defmethod mu_torsion-val ((m <SetSurfaceParams-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:mu_torsion-val is deprecated.  Use gazebo_ext_msgs-srv:mu_torsion instead.")
  (mu_torsion m))

(cl:ensure-generic-function 'patch_radius-val :lambda-list '(m))
(cl:defmethod patch_radius-val ((m <SetSurfaceParams-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:patch_radius-val is deprecated.  Use gazebo_ext_msgs-srv:patch_radius instead.")
  (patch_radius m))

(cl:ensure-generic-function 'poisson_ratio-val :lambda-list '(m))
(cl:defmethod poisson_ratio-val ((m <SetSurfaceParams-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:poisson_ratio-val is deprecated.  Use gazebo_ext_msgs-srv:poisson_ratio instead.")
  (poisson_ratio m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetSurfaceParams-request>) ostream)
  "Serializes a message object of type '<SetSurfaceParams-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'link_collision_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'link_collision_name))
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
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetSurfaceParams-request>) istream)
  "Deserializes a message object of type '<SetSurfaceParams-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'link_collision_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'link_collision_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
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
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetSurfaceParams-request>)))
  "Returns string type for a service object of type '<SetSurfaceParams-request>"
  "gazebo_ext_msgs/SetSurfaceParamsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSurfaceParams-request)))
  "Returns string type for a service object of type 'SetSurfaceParams-request"
  "gazebo_ext_msgs/SetSurfaceParamsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetSurfaceParams-request>)))
  "Returns md5sum for a message object of type '<SetSurfaceParams-request>"
  "5b42f161dd595c17f404172b095c2d93")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetSurfaceParams-request)))
  "Returns md5sum for a message object of type 'SetSurfaceParams-request"
  "5b42f161dd595c17f404172b095c2d93")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetSurfaceParams-request>)))
  "Returns full string definition for message of type '<SetSurfaceParams-request>"
  (cl:format cl:nil "string link_collision_name~%float64 elastic_modulus~%float64 mu1~%float64 mu2~%float64 mu_torsion~%float64 patch_radius~%float64 poisson_ratio~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetSurfaceParams-request)))
  "Returns full string definition for message of type 'SetSurfaceParams-request"
  (cl:format cl:nil "string link_collision_name~%float64 elastic_modulus~%float64 mu1~%float64 mu2~%float64 mu_torsion~%float64 patch_radius~%float64 poisson_ratio~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetSurfaceParams-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'link_collision_name))
     8
     8
     8
     8
     8
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetSurfaceParams-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetSurfaceParams-request
    (cl:cons ':link_collision_name (link_collision_name msg))
    (cl:cons ':elastic_modulus (elastic_modulus msg))
    (cl:cons ':mu1 (mu1 msg))
    (cl:cons ':mu2 (mu2 msg))
    (cl:cons ':mu_torsion (mu_torsion msg))
    (cl:cons ':patch_radius (patch_radius msg))
    (cl:cons ':poisson_ratio (poisson_ratio msg))
))
;//! \htmlinclude SetSurfaceParams-response.msg.html

(cl:defclass <SetSurfaceParams-response> (roslisp-msg-protocol:ros-message)
  ((success
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

(cl:defclass SetSurfaceParams-response (<SetSurfaceParams-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetSurfaceParams-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetSurfaceParams-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_ext_msgs-srv:<SetSurfaceParams-response> is deprecated: use gazebo_ext_msgs-srv:SetSurfaceParams-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SetSurfaceParams-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:success-val is deprecated.  Use gazebo_ext_msgs-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'status_message-val :lambda-list '(m))
(cl:defmethod status_message-val ((m <SetSurfaceParams-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:status_message-val is deprecated.  Use gazebo_ext_msgs-srv:status_message instead.")
  (status_message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetSurfaceParams-response>) ostream)
  "Serializes a message object of type '<SetSurfaceParams-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'status_message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'status_message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetSurfaceParams-response>) istream)
  "Deserializes a message object of type '<SetSurfaceParams-response>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetSurfaceParams-response>)))
  "Returns string type for a service object of type '<SetSurfaceParams-response>"
  "gazebo_ext_msgs/SetSurfaceParamsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSurfaceParams-response)))
  "Returns string type for a service object of type 'SetSurfaceParams-response"
  "gazebo_ext_msgs/SetSurfaceParamsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetSurfaceParams-response>)))
  "Returns md5sum for a message object of type '<SetSurfaceParams-response>"
  "5b42f161dd595c17f404172b095c2d93")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetSurfaceParams-response)))
  "Returns md5sum for a message object of type 'SetSurfaceParams-response"
  "5b42f161dd595c17f404172b095c2d93")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetSurfaceParams-response>)))
  "Returns full string definition for message of type '<SetSurfaceParams-response>"
  (cl:format cl:nil "bool success~%string status_message~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetSurfaceParams-response)))
  "Returns full string definition for message of type 'SetSurfaceParams-response"
  (cl:format cl:nil "bool success~%string status_message~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetSurfaceParams-response>))
  (cl:+ 0
     1
     4 (cl:length (cl:slot-value msg 'status_message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetSurfaceParams-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetSurfaceParams-response
    (cl:cons ':success (success msg))
    (cl:cons ':status_message (status_message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetSurfaceParams)))
  'SetSurfaceParams-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetSurfaceParams)))
  'SetSurfaceParams-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSurfaceParams)))
  "Returns string type for a service object of type '<SetSurfaceParams>"
  "gazebo_ext_msgs/SetSurfaceParams")