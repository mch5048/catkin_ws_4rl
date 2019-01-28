; Auto-generated. Do not edit!


(cl:in-package gazebo_ext_msgs-srv)


;//! \htmlinclude GetSkyProperties-request.msg.html

(cl:defclass <GetSkyProperties-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass GetSkyProperties-request (<GetSkyProperties-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetSkyProperties-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetSkyProperties-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_ext_msgs-srv:<GetSkyProperties-request> is deprecated: use gazebo_ext_msgs-srv:GetSkyProperties-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetSkyProperties-request>) ostream)
  "Serializes a message object of type '<GetSkyProperties-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetSkyProperties-request>) istream)
  "Deserializes a message object of type '<GetSkyProperties-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetSkyProperties-request>)))
  "Returns string type for a service object of type '<GetSkyProperties-request>"
  "gazebo_ext_msgs/GetSkyPropertiesRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSkyProperties-request)))
  "Returns string type for a service object of type 'GetSkyProperties-request"
  "gazebo_ext_msgs/GetSkyPropertiesRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetSkyProperties-request>)))
  "Returns md5sum for a message object of type '<GetSkyProperties-request>"
  "785496b76654a68787d4a95f6a2d2af4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetSkyProperties-request)))
  "Returns md5sum for a message object of type 'GetSkyProperties-request"
  "785496b76654a68787d4a95f6a2d2af4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetSkyProperties-request>)))
  "Returns full string definition for message of type '<GetSkyProperties-request>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetSkyProperties-request)))
  "Returns full string definition for message of type 'GetSkyProperties-request"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetSkyProperties-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetSkyProperties-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetSkyProperties-request
))
;//! \htmlinclude GetSkyProperties-response.msg.html

(cl:defclass <GetSkyProperties-response> (roslisp-msg-protocol:ros-message)
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

(cl:defclass GetSkyProperties-response (<GetSkyProperties-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetSkyProperties-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetSkyProperties-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name gazebo_ext_msgs-srv:<GetSkyProperties-response> is deprecated: use gazebo_ext_msgs-srv:GetSkyProperties-response instead.")))

(cl:ensure-generic-function 'time-val :lambda-list '(m))
(cl:defmethod time-val ((m <GetSkyProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:time-val is deprecated.  Use gazebo_ext_msgs-srv:time instead.")
  (time m))

(cl:ensure-generic-function 'sunrise-val :lambda-list '(m))
(cl:defmethod sunrise-val ((m <GetSkyProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:sunrise-val is deprecated.  Use gazebo_ext_msgs-srv:sunrise instead.")
  (sunrise m))

(cl:ensure-generic-function 'sunset-val :lambda-list '(m))
(cl:defmethod sunset-val ((m <GetSkyProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:sunset-val is deprecated.  Use gazebo_ext_msgs-srv:sunset instead.")
  (sunset m))

(cl:ensure-generic-function 'wind_speed-val :lambda-list '(m))
(cl:defmethod wind_speed-val ((m <GetSkyProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:wind_speed-val is deprecated.  Use gazebo_ext_msgs-srv:wind_speed instead.")
  (wind_speed m))

(cl:ensure-generic-function 'wind_direction-val :lambda-list '(m))
(cl:defmethod wind_direction-val ((m <GetSkyProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:wind_direction-val is deprecated.  Use gazebo_ext_msgs-srv:wind_direction instead.")
  (wind_direction m))

(cl:ensure-generic-function 'cloud_ambient-val :lambda-list '(m))
(cl:defmethod cloud_ambient-val ((m <GetSkyProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:cloud_ambient-val is deprecated.  Use gazebo_ext_msgs-srv:cloud_ambient instead.")
  (cloud_ambient m))

(cl:ensure-generic-function 'humidity-val :lambda-list '(m))
(cl:defmethod humidity-val ((m <GetSkyProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:humidity-val is deprecated.  Use gazebo_ext_msgs-srv:humidity instead.")
  (humidity m))

(cl:ensure-generic-function 'mean_cloud_size-val :lambda-list '(m))
(cl:defmethod mean_cloud_size-val ((m <GetSkyProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:mean_cloud_size-val is deprecated.  Use gazebo_ext_msgs-srv:mean_cloud_size instead.")
  (mean_cloud_size m))

(cl:ensure-generic-function 'success-val :lambda-list '(m))
(cl:defmethod success-val ((m <GetSkyProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:success-val is deprecated.  Use gazebo_ext_msgs-srv:success instead.")
  (success m))

(cl:ensure-generic-function 'status_message-val :lambda-list '(m))
(cl:defmethod status_message-val ((m <GetSkyProperties-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader gazebo_ext_msgs-srv:status_message-val is deprecated.  Use gazebo_ext_msgs-srv:status_message instead.")
  (status_message m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetSkyProperties-response>) ostream)
  "Serializes a message object of type '<GetSkyProperties-response>"
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
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'success) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'status_message))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'status_message))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetSkyProperties-response>) istream)
  "Deserializes a message object of type '<GetSkyProperties-response>"
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
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetSkyProperties-response>)))
  "Returns string type for a service object of type '<GetSkyProperties-response>"
  "gazebo_ext_msgs/GetSkyPropertiesResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSkyProperties-response)))
  "Returns string type for a service object of type 'GetSkyProperties-response"
  "gazebo_ext_msgs/GetSkyPropertiesResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetSkyProperties-response>)))
  "Returns md5sum for a message object of type '<GetSkyProperties-response>"
  "785496b76654a68787d4a95f6a2d2af4")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetSkyProperties-response)))
  "Returns md5sum for a message object of type 'GetSkyProperties-response"
  "785496b76654a68787d4a95f6a2d2af4")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetSkyProperties-response>)))
  "Returns full string definition for message of type '<GetSkyProperties-response>"
  (cl:format cl:nil "float64 time~%float64 sunrise~%float64 sunset~%float64 wind_speed~%float64 wind_direction~%std_msgs/ColorRGBA cloud_ambient~%float64 humidity~%float64 mean_cloud_size~%bool success~%string status_message~%~%~%================================================================================~%MSG: std_msgs/ColorRGBA~%float32 r~%float32 g~%float32 b~%float32 a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetSkyProperties-response)))
  "Returns full string definition for message of type 'GetSkyProperties-response"
  (cl:format cl:nil "float64 time~%float64 sunrise~%float64 sunset~%float64 wind_speed~%float64 wind_direction~%std_msgs/ColorRGBA cloud_ambient~%float64 humidity~%float64 mean_cloud_size~%bool success~%string status_message~%~%~%================================================================================~%MSG: std_msgs/ColorRGBA~%float32 r~%float32 g~%float32 b~%float32 a~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetSkyProperties-response>))
  (cl:+ 0
     8
     8
     8
     8
     8
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'cloud_ambient))
     8
     8
     1
     4 (cl:length (cl:slot-value msg 'status_message))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetSkyProperties-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetSkyProperties-response
    (cl:cons ':time (time msg))
    (cl:cons ':sunrise (sunrise msg))
    (cl:cons ':sunset (sunset msg))
    (cl:cons ':wind_speed (wind_speed msg))
    (cl:cons ':wind_direction (wind_direction msg))
    (cl:cons ':cloud_ambient (cloud_ambient msg))
    (cl:cons ':humidity (humidity msg))
    (cl:cons ':mean_cloud_size (mean_cloud_size msg))
    (cl:cons ':success (success msg))
    (cl:cons ':status_message (status_message msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetSkyProperties)))
  'GetSkyProperties-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetSkyProperties)))
  'GetSkyProperties-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetSkyProperties)))
  "Returns string type for a service object of type '<GetSkyProperties>"
  "gazebo_ext_msgs/GetSkyProperties")