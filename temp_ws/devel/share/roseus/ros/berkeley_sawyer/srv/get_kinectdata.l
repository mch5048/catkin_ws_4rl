;; Auto-generated. Do not edit!


(when (boundp 'berkeley_sawyer::get_kinectdata)
  (if (not (find-package "BERKELEY_SAWYER"))
    (make-package "BERKELEY_SAWYER"))
  (shadow 'get_kinectdata (find-package "BERKELEY_SAWYER")))
(unless (find-package "BERKELEY_SAWYER::GET_KINECTDATA")
  (make-package "BERKELEY_SAWYER::GET_KINECTDATA"))
(unless (find-package "BERKELEY_SAWYER::GET_KINECTDATAREQUEST")
  (make-package "BERKELEY_SAWYER::GET_KINECTDATAREQUEST"))
(unless (find-package "BERKELEY_SAWYER::GET_KINECTDATARESPONSE")
  (make-package "BERKELEY_SAWYER::GET_KINECTDATARESPONSE"))

(in-package "ROS")



(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))


(defclass berkeley_sawyer::get_kinectdataRequest
  :super ros::object
  :slots ())

(defmethod berkeley_sawyer::get_kinectdataRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass berkeley_sawyer::get_kinectdataResponse
  :super ros::object
  :slots (_image ))

(defmethod berkeley_sawyer::get_kinectdataResponse
  (:init
   (&key
    ((:image __image) (instance sensor_msgs::Image :init))
    )
   (send-super :init)
   (setq _image __image)
   self)
  (:image
   (&rest __image)
   (if (keywordp (car __image))
       (send* _image __image)
     (progn
       (if __image (setq _image (car __image)))
       _image)))
  (:serialization-length
   ()
   (+
    ;; sensor_msgs/Image _image
    (send _image :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; sensor_msgs/Image _image
       (send _image :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; sensor_msgs/Image _image
     (send _image :deserialize buf ptr-) (incf ptr- (send _image :serialization-length))
   ;;
   self)
  )

(defclass berkeley_sawyer::get_kinectdata
  :super ros::object
  :slots ())

(setf (get berkeley_sawyer::get_kinectdata :md5sum-) "b13d2865c5af2a64e6e30ab1b56e1dd5")
(setf (get berkeley_sawyer::get_kinectdata :datatype-) "berkeley_sawyer/get_kinectdata")
(setf (get berkeley_sawyer::get_kinectdata :request) berkeley_sawyer::get_kinectdataRequest)
(setf (get berkeley_sawyer::get_kinectdata :response) berkeley_sawyer::get_kinectdataResponse)

(defmethod berkeley_sawyer::get_kinectdataRequest
  (:response () (instance berkeley_sawyer::get_kinectdataResponse :init)))

(setf (get berkeley_sawyer::get_kinectdataRequest :md5sum-) "b13d2865c5af2a64e6e30ab1b56e1dd5")
(setf (get berkeley_sawyer::get_kinectdataRequest :datatype-) "berkeley_sawyer/get_kinectdataRequest")
(setf (get berkeley_sawyer::get_kinectdataRequest :definition-)
      "---
sensor_msgs/Image image

================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of cameara
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
# 0: no frame
# 1: global frame
string frame_id
")

(setf (get berkeley_sawyer::get_kinectdataResponse :md5sum-) "b13d2865c5af2a64e6e30ab1b56e1dd5")
(setf (get berkeley_sawyer::get_kinectdataResponse :datatype-) "berkeley_sawyer/get_kinectdataResponse")
(setf (get berkeley_sawyer::get_kinectdataResponse :definition-)
      "---
sensor_msgs/Image image

================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of cameara
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
# 0: no frame
# 1: global frame
string frame_id
")



(provide :berkeley_sawyer/get_kinectdata "b13d2865c5af2a64e6e30ab1b56e1dd5")


