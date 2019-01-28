; Auto-generated. Do not edit!


(cl:in-package object_detection_yolov2-msg)


;//! \htmlinclude DetectionFull.msg.html

(cl:defclass <DetectionFull> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (image
    :reader image
    :initarg :image
    :type sensor_msgs-msg:Image
    :initform (cl:make-instance 'sensor_msgs-msg:Image))
   (masks
    :reader masks
    :initarg :masks
    :type (cl:vector sensor_msgs-msg:Image)
   :initform (cl:make-array 0 :element-type 'sensor_msgs-msg:Image :initial-element (cl:make-instance 'sensor_msgs-msg:Image)))
   (detections
    :reader detections
    :initarg :detections
    :type object_detection_yolov2-msg:DetectionArray
    :initform (cl:make-instance 'object_detection_yolov2-msg:DetectionArray)))
)

(cl:defclass DetectionFull (<DetectionFull>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <DetectionFull>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'DetectionFull)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name object_detection_yolov2-msg:<DetectionFull> is deprecated: use object_detection_yolov2-msg:DetectionFull instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <DetectionFull>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_detection_yolov2-msg:header-val is deprecated.  Use object_detection_yolov2-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'image-val :lambda-list '(m))
(cl:defmethod image-val ((m <DetectionFull>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_detection_yolov2-msg:image-val is deprecated.  Use object_detection_yolov2-msg:image instead.")
  (image m))

(cl:ensure-generic-function 'masks-val :lambda-list '(m))
(cl:defmethod masks-val ((m <DetectionFull>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_detection_yolov2-msg:masks-val is deprecated.  Use object_detection_yolov2-msg:masks instead.")
  (masks m))

(cl:ensure-generic-function 'detections-val :lambda-list '(m))
(cl:defmethod detections-val ((m <DetectionFull>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader object_detection_yolov2-msg:detections-val is deprecated.  Use object_detection_yolov2-msg:detections instead.")
  (detections m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <DetectionFull>) ostream)
  "Serializes a message object of type '<DetectionFull>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'image) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'masks))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'masks))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'detections) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <DetectionFull>) istream)
  "Deserializes a message object of type '<DetectionFull>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'image) istream)
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'masks) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'masks)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'sensor_msgs-msg:Image))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'detections) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<DetectionFull>)))
  "Returns string type for a message object of type '<DetectionFull>"
  "object_detection_yolov2/DetectionFull")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'DetectionFull)))
  "Returns string type for a message object of type 'DetectionFull"
  "object_detection_yolov2/DetectionFull")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<DetectionFull>)))
  "Returns md5sum for a message object of type '<DetectionFull>"
  "3b39abf49a96981c609db709bdd09c4d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'DetectionFull)))
  "Returns md5sum for a message object of type 'DetectionFull"
  "3b39abf49a96981c609db709bdd09c4d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<DetectionFull>)))
  "Returns full string definition for message of type '<DetectionFull>"
  (cl:format cl:nil "Header header~%~%# The image containing the detetions~%sensor_msgs/Image image~%~%# binary images containing masks~%sensor_msgs/Image[] masks~%~%# The array containing all the detections~%DetectionArray detections~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of cameara~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: object_detection_yolov2/DetectionArray~%Header header~%~%# The size of the array~%uint32 size~%# The array containing all the detections~%Detection[] data~%~%================================================================================~%MSG: object_detection_yolov2/Detection~%Header header~%~%string object_class~%float32 p~%~%uint16 x~%uint16 y~%~%float32 cam_x~%float32 cam_y~%float32 cam_z~%~%uint16 width~%uint16 height~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'DetectionFull)))
  "Returns full string definition for message of type 'DetectionFull"
  (cl:format cl:nil "Header header~%~%# The image containing the detetions~%sensor_msgs/Image image~%~%# binary images containing masks~%sensor_msgs/Image[] masks~%~%# The array containing all the detections~%DetectionArray detections~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of cameara~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%================================================================================~%MSG: object_detection_yolov2/DetectionArray~%Header header~%~%# The size of the array~%uint32 size~%# The array containing all the detections~%Detection[] data~%~%================================================================================~%MSG: object_detection_yolov2/Detection~%Header header~%~%string object_class~%float32 p~%~%uint16 x~%uint16 y~%~%float32 cam_x~%float32 cam_y~%float32 cam_z~%~%uint16 width~%uint16 height~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <DetectionFull>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'image))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'masks) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'detections))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <DetectionFull>))
  "Converts a ROS message object to a list"
  (cl:list 'DetectionFull
    (cl:cons ':header (header msg))
    (cl:cons ':image (image msg))
    (cl:cons ':masks (masks msg))
    (cl:cons ':detections (detections msg))
))
