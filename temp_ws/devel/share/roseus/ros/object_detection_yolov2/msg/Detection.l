;; Auto-generated. Do not edit!


(when (boundp 'object_detection_yolov2::Detection)
  (if (not (find-package "OBJECT_DETECTION_YOLOV2"))
    (make-package "OBJECT_DETECTION_YOLOV2"))
  (shadow 'Detection (find-package "OBJECT_DETECTION_YOLOV2")))
(unless (find-package "OBJECT_DETECTION_YOLOV2::DETECTION")
  (make-package "OBJECT_DETECTION_YOLOV2::DETECTION"))

(in-package "ROS")
;;//! \htmlinclude Detection.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass object_detection_yolov2::Detection
  :super ros::object
  :slots (_header _object_class _p _x _y _cam_x _cam_y _cam_z _width _height ))

(defmethod object_detection_yolov2::Detection
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:object_class __object_class) "")
    ((:p __p) 0.0)
    ((:x __x) 0)
    ((:y __y) 0)
    ((:cam_x __cam_x) 0.0)
    ((:cam_y __cam_y) 0.0)
    ((:cam_z __cam_z) 0.0)
    ((:width __width) 0)
    ((:height __height) 0)
    )
   (send-super :init)
   (setq _header __header)
   (setq _object_class (string __object_class))
   (setq _p (float __p))
   (setq _x (round __x))
   (setq _y (round __y))
   (setq _cam_x (float __cam_x))
   (setq _cam_y (float __cam_y))
   (setq _cam_z (float __cam_z))
   (setq _width (round __width))
   (setq _height (round __height))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:object_class
   (&optional __object_class)
   (if __object_class (setq _object_class __object_class)) _object_class)
  (:p
   (&optional __p)
   (if __p (setq _p __p)) _p)
  (:x
   (&optional __x)
   (if __x (setq _x __x)) _x)
  (:y
   (&optional __y)
   (if __y (setq _y __y)) _y)
  (:cam_x
   (&optional __cam_x)
   (if __cam_x (setq _cam_x __cam_x)) _cam_x)
  (:cam_y
   (&optional __cam_y)
   (if __cam_y (setq _cam_y __cam_y)) _cam_y)
  (:cam_z
   (&optional __cam_z)
   (if __cam_z (setq _cam_z __cam_z)) _cam_z)
  (:width
   (&optional __width)
   (if __width (setq _width __width)) _width)
  (:height
   (&optional __height)
   (if __height (setq _height __height)) _height)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; string _object_class
    4 (length _object_class)
    ;; float32 _p
    4
    ;; uint16 _x
    2
    ;; uint16 _y
    2
    ;; float32 _cam_x
    4
    ;; float32 _cam_y
    4
    ;; float32 _cam_z
    4
    ;; uint16 _width
    2
    ;; uint16 _height
    2
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; string _object_class
       (write-long (length _object_class) s) (princ _object_class s)
     ;; float32 _p
       (sys::poke _p (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; uint16 _x
       (write-word _x s)
     ;; uint16 _y
       (write-word _y s)
     ;; float32 _cam_x
       (sys::poke _cam_x (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _cam_y
       (sys::poke _cam_y (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _cam_z
       (sys::poke _cam_z (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; uint16 _width
       (write-word _width s)
     ;; uint16 _height
       (write-word _height s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; string _object_class
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _object_class (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float32 _p
     (setq _p (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; uint16 _x
     (setq _x (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _y
     (setq _y (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; float32 _cam_x
     (setq _cam_x (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _cam_y
     (setq _cam_y (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _cam_z
     (setq _cam_z (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; uint16 _width
     (setq _width (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _height
     (setq _height (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;;
   self)
  )

(setf (get object_detection_yolov2::Detection :md5sum-) "d82132341465f8c6318faea203b0884c")
(setf (get object_detection_yolov2::Detection :datatype-) "object_detection_yolov2/Detection")
(setf (get object_detection_yolov2::Detection :definition-)
      "Header header

string object_class
float32 p

uint16 x
uint16 y

float32 cam_x
float32 cam_y
float32 cam_z

uint16 width
uint16 height

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



(provide :object_detection_yolov2/Detection "d82132341465f8c6318faea203b0884c")


