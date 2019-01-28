; Auto-generated. Do not edit!


(cl:in-package gazebo_ext_msgs-srv)


;//! \htmlinclude SetLinkVisualProperties-request.msg.html

(cl:defclass <SetLinkVisualProperties-request> (roslisp-msg-protocol:ros-message)
  ((link_visual_name
    :reader link_visual_name
    :initarg :link_visual_name
    :type cl:string
    :initform "")
   (ambient
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
    :initform (cl:make-instance 'std_msgs-msg:ColorRGBA)))
)

(cl:defclass SetLinkVisualProperties-request (<SetLinkVisualProperties-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetLinkVisualProperties-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetLinkVisualProperties-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_ext_msgs-srv:<SetLinkVisualProperties-request> is deprecated: use gazebo_ext_msgs-srv:SetLinkVisualProperties-request instead.")))

(cl:ensure-generic-function 'link_visual_name-val :lambda-list '(m))
(cl:defmethod link_visual_name-val ((m <SetLinkVisualProperties-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:link_visual_name-val is deprecated.  Use gazebo_ext_msgs-srv:link_visual_name instead.")
  (link_visual_name m))

(cl:ensure-generic-function 'ambient-val :lambda-list '(m))
(cl:defmethod ambient-val ((m <SetLinkVisualProperties-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:ambient-val is deprecated.  Use gazebo_ext_msgs-srv:ambient instead.")
  (ambient m))

(cl:ensure-generic-function 'diffuse-val :lambda-list '(m))
(cl:defmethod diffuse-val ((m <SetLinkVisualProperties-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:diffuse-val is deprecated.  Use gazebo_ext_msgs-srv:diffuse instead.")
  (diffuse m))

(cl:ensure-generic-function 'specular-val :lambda-list '(m))
(cl:defmethod specular-val ((m <SetLinkVisualProperties-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:specular-val is deprecated.  Use gazebo_ext_msgs-srv:specular instead.")
  (specular m))

(cl:ensure-generic-function 'emissive-val :lambda-list '(m))
(cl:defmethod emissive-val ((m <SetLinkVisualProperties-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:emissive-val is deprecated.  Use gazebo_ext_msgs-srv:emissive instead.")
  (emissive m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetLinkVisualProperties-request>) ostream)
  "Serializes a message object of type '<SetLinkVisualProperties-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'link_visual_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'link_visual_name))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'ambient) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'diffuse) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'specular) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'emissive) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetLinkVisualProperties-request>) istream)
  "Deserializes a message object of type '<SetLinkVisualProperties-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'link_visual_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'link_visual_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'ambient) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'diffuse) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'specular) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'emissive) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetLinkVisualProperties-request>)))
  "Returns string type for a service object of type '<SetLinkVisualProperties-request>"
  "gazebo_ext_msgs/SetLinkVisualPropertiesRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetLinkVisualProperties-request)))
  "Returns string type for a service object of type 'SetLinkVisualProperties-request"
  "gazebo_ext_msgs/SetLinkVisualPropertiesRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetLinkVisualProperties-request>)))
  "Returns md5sum for a message object of type '<SetLinkVisualProperties-request>"
  "defa425a32f63c1cdc8261ea0e650ab9")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetLinkVisualProperties-request)))
  "Returns md5sum for a message object of type 'SetLinkVisualProperties-request"
  "defa425a32f63c1cdc8261ea0e650ab9")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetLinkVisualProperties-request>)))
  "Returns full string definition for message of type '<SetLinkVisualProperties-request>"
  (cl:format cl:nil "string link_visual_name~%std_msgs/ColorRGBA ambient~%std_msgs/ColorRGBA diffuse~%std_msgs/ColorRGBA specular~%std_msgs/ColorRGBA emissive~%~%================================================================================~%MSG: std_msgs/ColorRGBA~%float32 r~%float32 g~%float32 b~%float32 a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetLinkVisualProperties-request)))
  "Returns full string definition for message of type 'SetLinkVisualProperties-request"
  (cl:format cl:nil "string link_visual_name~%std_msgs/ColorRGBA ambient~%std_msgs/ColorRGBA diffuse~%std_msgs/ColorRGBA specular~%std_msgs/ColorRGBA emissive~%~%================================================================================~%MSG: std_msgs/ColorRGBA~%float32 r~%float32 g~%float32 b~%float32 a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetLinkVisualProperties-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'link_visual_name))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'ambient))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'diffuse))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'specular))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'emissive))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetLinkVisualProperties-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetLinkVisualProperties-request
    (cl:cons ':link_visual_name (link_visual_name msg))
    (cl:cons ':ambient (ambient msg))
    (cl:cons ':diffuse (diffuse msg))
    (cl:cons ':specular (specular msg))
    (cl:cons ':emissive (emissive msg))
))
;//! \htmlinclude SetLinkVisualProperties-response.msg.html

(cl:defclass <SetLinkVisualProperties-response> (roslisp-msg-protocol:ros-message)
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

(cl:defclass SetLinkVisualProperties-response (<SetLinkVisualProperties-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetLinkVisualProperties-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetLinkVisualProperties-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_ext_msgs-srv:<SetLinkVisualProperties-response> is deprecated: use gazebo_ext_msgs-srv:SetLinkVisualProperties-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SetLinkVisualProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:success-val is deprecated.  Use gazebo_ext_msgs-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'status_message-val :lambda-list '(m))
(cl:defmethod status_message-val ((m <SetLinkVisualProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:status_message-val is deprecated.  Use gazebo_ext_msgs-srv:status_message instead.")
  (status_message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetLinkVisualProperties-response>) ostream)
  "Serializes a message object of type '<SetLinkVisualProperties-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'status_message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'status_message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetLinkVisualProperties-response>) istream)
  "Deserializes a message object of type '<SetLinkVisualProperties-response>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetLinkVisualProperties-response>)))
  "Returns string type for a service object of type '<SetLinkVisualProperties-response>"
  "gazebo_ext_msgs/SetLinkVisualPropertiesResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetLinkVisualProperties-response)))
  "Returns string type for a service object of type 'SetLinkVisualProperties-response"
  "gazebo_ext_msgs/SetLinkVisualPropertiesResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetLinkVisualProperties-response>)))
  "Returns md5sum for a message object of type '<SetLinkVisualProperties-response>"
  "defa425a32f63c1cdc8261ea0e650ab9")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetLinkVisualProperties-response)))
  "Returns md5sum for a message object of type 'SetLinkVisualProperties-response"
  "defa425a32f63c1cdc8261ea0e650ab9")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetLinkVisualProperties-response>)))
  "Returns full string definition for message of type '<SetLinkVisualProperties-response>"
  (cl:format cl:nil "bool success~%string status_message~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetLinkVisualProperties-response)))
  "Returns full string definition for message of type 'SetLinkVisualProperties-response"
  (cl:format cl:nil "bool success~%string status_message~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetLinkVisualProperties-response>))
  (cl:+ 0
     1
     4 (cl:length (cl:slot-value msg 'status_message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetLinkVisualProperties-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetLinkVisualProperties-response
    (cl:cons ':success (success msg))
    (cl:cons ':status_message (status_message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetLinkVisualProperties)))
  'SetLinkVisualProperties-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetLinkVisualProperties)))
  'SetLinkVisualProperties-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetLinkVisualProperties)))
  "Returns string type for a service object of type '<SetLinkVisualProperties>"
  "gazebo_ext_msgs/SetLinkVisualProperties")