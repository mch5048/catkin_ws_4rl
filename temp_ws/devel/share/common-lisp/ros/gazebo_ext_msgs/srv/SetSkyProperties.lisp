; Auto-generated. Do not edit!


(cl:in-package gazebo_ext_msgs-srv)


;//! \htmlinclude SetSkyProperties-request.msg.html

(cl:defclass <SetSkyProperties-request> (roslisp-msg-protocol:ros-message)
  ((time
    :reader time
    :initarg :time
    :type cl:float
    :initform 0.0)
   (sunrise
    :reader sunrise
    :initarg :sunrise
    :type cl:float
    :initform 0.0)
   (sunset
    :reader sunset
    :initarg :sunset
    :type cl:float
    :initform 0.0)
   (wind_speed
    :reader wind_speed
    :initarg :wind_speed
    :type cl:float
    :initform 0.0)
   (wind_direction
    :reader wind_direction
    :initarg :wind_direction
    :type cl:float
    :initform 0.0)
   (cloud_ambient
    :reader cloud_ambient
    :initarg :cloud_ambient
    :type std_msgs-msg:ColorRGBA
    :initform (cl:make-instance 'std_msgs-msg:ColorRGBA))
   (humidity
    :reader humidity
    :initarg :humidity
    :type cl:float
    :initform 0.0)
   (mean_cloud_size
    :reader mean_cloud_size
    :initarg :mean_cloud_size
    :type cl:float
    :initform 0.0))
)

(cl:defclass SetSkyProperties-request (<SetSkyProperties-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetSkyProperties-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetSkyProperties-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_ext_msgs-srv:<SetSkyProperties-request> is deprecated: use gazebo_ext_msgs-srv:SetSkyProperties-request instead.")))

(cl:ensure-generic-function 'time-val :lambda-list '(m))
(cl:defmethod time-val ((m <SetSkyProperties-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:time-val is deprecated.  Use gazebo_ext_msgs-srv:time instead.")
  (time m))

(cl:ensure-generic-function 'sunrise-val :lambda-list '(m))
(cl:defmethod sunrise-val ((m <SetSkyProperties-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:sunrise-val is deprecated.  Use gazebo_ext_msgs-srv:sunrise instead.")
  (sunrise m))

(cl:ensure-generic-function 'sunset-val :lambda-list '(m))
(cl:defmethod sunset-val ((m <SetSkyProperties-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:sunset-val is deprecated.  Use gazebo_ext_msgs-srv:sunset instead.")
  (sunset m))

(cl:ensure-generic-function 'wind_speed-val :lambda-list '(m))
(cl:defmethod wind_speed-val ((m <SetSkyProperties-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:wind_speed-val is deprecated.  Use gazebo_ext_msgs-srv:wind_speed instead.")
  (wind_speed m))

(cl:ensure-generic-function 'wind_direction-val :lambda-list '(m))
(cl:defmethod wind_direction-val ((m <SetSkyProperties-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:wind_direction-val is deprecated.  Use gazebo_ext_msgs-srv:wind_direction instead.")
  (wind_direction m))

(cl:ensure-generic-function 'cloud_ambient-val :lambda-list '(m))
(cl:defmethod cloud_ambient-val ((m <SetSkyProperties-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:cloud_ambient-val is deprecated.  Use gazebo_ext_msgs-srv:cloud_ambient instead.")
  (cloud_ambient m))

(cl:ensure-generic-function 'humidity-val :lambda-list '(m))
(cl:defmethod humidity-val ((m <SetSkyProperties-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:humidity-val is deprecated.  Use gazebo_ext_msgs-srv:humidity instead.")
  (humidity m))

(cl:ensure-generic-function 'mean_cloud_size-val :lambda-list '(m))
(cl:defmethod mean_cloud_size-val ((m <SetSkyProperties-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:mean_cloud_size-val is deprecated.  Use gazebo_ext_msgs-srv:mean_cloud_size instead.")
  (mean_cloud_size m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetSkyProperties-request>) ostream)
  "Serializes a message object of type '<SetSkyProperties-request>"
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'time))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'sunrise))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'sunset))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'wind_speed))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'wind_direction))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'cloud_ambient) ostream)
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'humidity))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-double-float-bits (cl:slot-value msg 'mean_cloud_size))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 32) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 40) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 48) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 56) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetSkyProperties-request>) istream)
  "Deserializes a message object of type '<SetSkyProperties-request>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'time) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'sunrise) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'sunset) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'wind_speed) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'wind_direction) (roslisp-utils:decode-double-float-bits bits)))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'cloud_ambient) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'humidity) (roslisp-utils:decode-double-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 32) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 40) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 48) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 56) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'mean_cloud_size) (roslisp-utils:decode-double-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetSkyProperties-request>)))
  "Returns string type for a service object of type '<SetSkyProperties-request>"
  "gazebo_ext_msgs/SetSkyPropertiesRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSkyProperties-request)))
  "Returns string type for a service object of type 'SetSkyProperties-request"
  "gazebo_ext_msgs/SetSkyPropertiesRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetSkyProperties-request>)))
  "Returns md5sum for a message object of type '<SetSkyProperties-request>"
  "58ab80a272655f0012daa4ccd6a59539")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetSkyProperties-request)))
  "Returns md5sum for a message object of type 'SetSkyProperties-request"
  "58ab80a272655f0012daa4ccd6a59539")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetSkyProperties-request>)))
  "Returns full string definition for message of type '<SetSkyProperties-request>"
  (cl:format cl:nil "float64 time~%float64 sunrise~%float64 sunset~%float64 wind_speed~%float64 wind_direction~%std_msgs/ColorRGBA cloud_ambient~%float64 humidity~%float64 mean_cloud_size~%~%================================================================================~%MSG: std_msgs/ColorRGBA~%float32 r~%float32 g~%float32 b~%float32 a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetSkyProperties-request)))
  "Returns full string definition for message of type 'SetSkyProperties-request"
  (cl:format cl:nil "float64 time~%float64 sunrise~%float64 sunset~%float64 wind_speed~%float64 wind_direction~%std_msgs/ColorRGBA cloud_ambient~%float64 humidity~%float64 mean_cloud_size~%~%================================================================================~%MSG: std_msgs/ColorRGBA~%float32 r~%float32 g~%float32 b~%float32 a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetSkyProperties-request>))
  (cl:+ 0
     8
     8
     8
     8
     8
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'cloud_ambient))
     8
     8
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetSkyProperties-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetSkyProperties-request
    (cl:cons ':time (time msg))
    (cl:cons ':sunrise (sunrise msg))
    (cl:cons ':sunset (sunset msg))
    (cl:cons ':wind_speed (wind_speed msg))
    (cl:cons ':wind_direction (wind_direction msg))
    (cl:cons ':cloud_ambient (cloud_ambient msg))
    (cl:cons ':humidity (humidity msg))
    (cl:cons ':mean_cloud_size (mean_cloud_size msg))
))
;//! \htmlinclude SetSkyProperties-response.msg.html

(cl:defclass <SetSkyProperties-response> (roslisp-msg-protocol:ros-message)
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

(cl:defclass SetSkyProperties-response (<SetSkyProperties-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetSkyProperties-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetSkyProperties-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_ext_msgs-srv:<SetSkyProperties-response> is deprecated: use gazebo_ext_msgs-srv:SetSkyProperties-response instead.")))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <SetSkyProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:success-val is deprecated.  Use gazebo_ext_msgs-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'status_message-val :lambda-list '(m))
(cl:defmethod status_message-val ((m <SetSkyProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:status_message-val is deprecated.  Use gazebo_ext_msgs-srv:status_message instead.")
  (status_message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetSkyProperties-response>) ostream)
  "Serializes a message object of type '<SetSkyProperties-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'status_message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'status_message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetSkyProperties-response>) istream)
  "Deserializes a message object of type '<SetSkyProperties-response>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetSkyProperties-response>)))
  "Returns string type for a service object of type '<SetSkyProperties-response>"
  "gazebo_ext_msgs/SetSkyPropertiesResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSkyProperties-response)))
  "Returns string type for a service object of type 'SetSkyProperties-response"
  "gazebo_ext_msgs/SetSkyPropertiesResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetSkyProperties-response>)))
  "Returns md5sum for a message object of type '<SetSkyProperties-response>"
  "58ab80a272655f0012daa4ccd6a59539")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetSkyProperties-response)))
  "Returns md5sum for a message object of type 'SetSkyProperties-response"
  "58ab80a272655f0012daa4ccd6a59539")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetSkyProperties-response>)))
  "Returns full string definition for message of type '<SetSkyProperties-response>"
  (cl:format cl:nil "bool success~%string status_message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetSkyProperties-response)))
  "Returns full string definition for message of type 'SetSkyProperties-response"
  (cl:format cl:nil "bool success~%string status_message~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetSkyProperties-response>))
  (cl:+ 0
     1
     4 (cl:length (cl:slot-value msg 'status_message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetSkyProperties-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetSkyProperties-response
    (cl:cons ':success (success msg))
    (cl:cons ':status_message (status_message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetSkyProperties)))
  'SetSkyProperties-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetSkyProperties)))
  'SetSkyProperties-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSkyProperties)))
  "Returns string type for a service object of type '<SetSkyProperties>"
  "gazebo_ext_msgs/SetSkyProperties")