; Auto-generated. Do not edit!


(cl:in-package gazebo_ext_msgs-srv)


;//! \htmlinclude GetLinkVisualProperties-request.msg.html

(cl:defclass <GetLinkVisualProperties-request> (roslisp-msg-protocol:ros-message)
  ((link_visual_name
    :reader link_visual_name
    :initarg :link_visual_name
    :type cl:string
    :initform ""))
)

(cl:defclass GetLinkVisualProperties-request (<GetLinkVisualProperties-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetLinkVisualProperties-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetLinkVisualProperties-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_ext_msgs-srv:<GetLinkVisualProperties-request> is deprecated: use gazebo_ext_msgs-srv:GetLinkVisualProperties-request instead.")))

(cl:ensure-generic-function 'link_visual_name-val :lambda-list '(m))
(cl:defmethod link_visual_name-val ((m <GetLinkVisualProperties-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:link_visual_name-val is deprecated.  Use gazebo_ext_msgs-srv:link_visual_name instead.")
  (link_visual_name m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetLinkVisualProperties-request>) ostream)
  "Serializes a message object of type '<GetLinkVisualProperties-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'link_visual_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'link_visual_name))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetLinkVisualProperties-request>) istream)
  "Deserializes a message object of type '<GetLinkVisualProperties-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'link_visual_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'link_visual_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetLinkVisualProperties-request>)))
  "Returns string type for a service object of type '<GetLinkVisualProperties-request>"
  "gazebo_ext_msgs/GetLinkVisualPropertiesRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetLinkVisualProperties-request)))
  "Returns string type for a service object of type 'GetLinkVisualProperties-request"
  "gazebo_ext_msgs/GetLinkVisualPropertiesRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetLinkVisualProperties-request>)))
  "Returns md5sum for a message object of type '<GetLinkVisualProperties-request>"
  "565eef77cf4ad97635bdc8bf4af90f87")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetLinkVisualProperties-request)))
  "Returns md5sum for a message object of type 'GetLinkVisualProperties-request"
  "565eef77cf4ad97635bdc8bf4af90f87")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetLinkVisualProperties-request>)))
  "Returns full string definition for message of type '<GetLinkVisualProperties-request>"
  (cl:format cl:nil "string link_visual_name~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetLinkVisualProperties-request)))
  "Returns full string definition for message of type 'GetLinkVisualProperties-request"
  (cl:format cl:nil "string link_visual_name~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetLinkVisualProperties-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'link_visual_name))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetLinkVisualProperties-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetLinkVisualProperties-request
    (cl:cons ':link_visual_name (link_visual_name msg))
))
;//! \htmlinclude GetLinkVisualProperties-response.msg.html

(cl:defclass <GetLinkVisualProperties-response> (roslisp-msg-protocol:ros-message)
  ((ambient
    :reader ambient
    :initarg :ambient
    :type std_msgs-msg:ColorRGBA
    :initform (cl:make-instance 'std_msgs-msg:ColorRGBA))
   (diffuse
    :reader diffuse
    :initarg :diffuse
    :type std_msgs-msg:ColorRGBA
    :initform (cl:make-instance 'std_msgs-msg:ColorRGBA))
   (specular
    :reader specular
    :initarg :specular
    :type std_msgs-msg:ColorRGBA
    :initform (cl:make-instance 'std_msgs-msg:ColorRGBA))
   (emissive
    :reader emissive
    :initarg :emissive
    :type std_msgs-msg:ColorRGBA
    :initform (cl:make-instance 'std_msgs-msg:ColorRGBA))
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

(cl:defclass GetLinkVisualProperties-response (<GetLinkVisualProperties-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetLinkVisualProperties-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetLinkVisualProperties-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_ext_msgs-srv:<GetLinkVisualProperties-response> is deprecated: use gazebo_ext_msgs-srv:GetLinkVisualProperties-response instead.")))

(cl:ensure-generic-function 'ambient-val :lambda-list '(m))
(cl:defmethod ambient-val ((m <GetLinkVisualProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:ambient-val is deprecated.  Use gazebo_ext_msgs-srv:ambient instead.")
  (ambient m))

(cl:ensure-generic-function 'diffuse-val :lambda-list '(m))
(cl:defmethod diffuse-val ((m <GetLinkVisualProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:diffuse-val is deprecated.  Use gazebo_ext_msgs-srv:diffuse instead.")
  (diffuse m))

(cl:ensure-generic-function 'specular-val :lambda-list '(m))
(cl:defmethod specular-val ((m <GetLinkVisualProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:specular-val is deprecated.  Use gazebo_ext_msgs-srv:specular instead.")
  (specular m))

(cl:ensure-generic-function 'emissive-val :lambda-list '(m))
(cl:defmethod emissive-val ((m <GetLinkVisualProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:emissive-val is deprecated.  Use gazebo_ext_msgs-srv:emissive instead.")
  (emissive m))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <GetLinkVisualProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:success-val is deprecated.  Use gazebo_ext_msgs-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'status_message-val :lambda-list '(m))
(cl:defmethod status_message-val ((m <GetLinkVisualProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:status_message-val is deprecated.  Use gazebo_ext_msgs-srv:status_message instead.")
  (status_message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetLinkVisualProperties-response>) ostream)
  "Serializes a message object of type '<GetLinkVisualProperties-response>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'ambient) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'diffuse) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'specular) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'emissive) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'status_message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'status_message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetLinkVisualProperties-response>) istream)
  "Deserializes a message object of type '<GetLinkVisualProperties-response>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'ambient) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'diffuse) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'specular) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'emissive) istream)
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetLinkVisualProperties-response>)))
  "Returns string type for a service object of type '<GetLinkVisualProperties-response>"
  "gazebo_ext_msgs/GetLinkVisualPropertiesResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetLinkVisualProperties-response)))
  "Returns string type for a service object of type 'GetLinkVisualProperties-response"
  "gazebo_ext_msgs/GetLinkVisualPropertiesResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetLinkVisualProperties-response>)))
  "Returns md5sum for a message object of type '<GetLinkVisualProperties-response>"
  "565eef77cf4ad97635bdc8bf4af90f87")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetLinkVisualProperties-response)))
  "Returns md5sum for a message object of type 'GetLinkVisualProperties-response"
  "565eef77cf4ad97635bdc8bf4af90f87")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetLinkVisualProperties-response>)))
  "Returns full string definition for message of type '<GetLinkVisualProperties-response>"
  (cl:format cl:nil "std_msgs/ColorRGBA ambient~%std_msgs/ColorRGBA diffuse~%std_msgs/ColorRGBA specular~%std_msgs/ColorRGBA emissive~%bool success~%string status_message~%~%================================================================================~%MSG: std_msgs/ColorRGBA~%float32 r~%float32 g~%float32 b~%float32 a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetLinkVisualProperties-response)))
  "Returns full string definition for message of type 'GetLinkVisualProperties-response"
  (cl:format cl:nil "std_msgs/ColorRGBA ambient~%std_msgs/ColorRGBA diffuse~%std_msgs/ColorRGBA specular~%std_msgs/ColorRGBA emissive~%bool success~%string status_message~%~%================================================================================~%MSG: std_msgs/ColorRGBA~%float32 r~%float32 g~%float32 b~%float32 a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetLinkVisualProperties-response>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'ambient))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'diffuse))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'specular))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'emissive))
     1
     4 (cl:length (cl:slot-value msg 'status_message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetLinkVisualProperties-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetLinkVisualProperties-response
    (cl:cons ':ambient (ambient msg))
    (cl:cons ':diffuse (diffuse msg))
    (cl:cons ':specular (specular msg))
    (cl:cons ':emissive (emissive msg))
    (cl:cons ':success (success msg))
    (cl:cons ':status_message (status_message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetLinkVisualProperties)))
  'GetLinkVisualProperties-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetLinkVisualProperties)))
  'GetLinkVisualProperties-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetLinkVisualProperties)))
  "Returns string type for a service object of type '<GetLinkVisualProperties>"
  "gazebo_ext_msgs/GetLinkVisualProperties")