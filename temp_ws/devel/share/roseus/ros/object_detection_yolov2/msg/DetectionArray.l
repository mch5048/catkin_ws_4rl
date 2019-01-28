;; Auto-generated. Do not edit!


(when (boundp 'object_detection_yolov2::DetectionArray)
  (if (not (find-package "OBJECT_DETECTION_YOLOV2"))
    (make-package "OBJECT_DETECTION_YOLOV2"))
  (shadow 'DetectionArray (find-package "OBJECT_DETECTION_YOLOV2")))
(unless (find-package "OBJECT_DETECTION_YOLOV2::DETECTIONARRAY")
  (make-package "OBJECT_DETECTION_YOLOV2::DETECTIONARRAY"))

(in-package "ROS")
;;//! \htmlinclude DetectionArray.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass object_detection_yolov2::DetectionArray
  :super ros::object
  :slots (_header _size _data ))

(defmethod object_detection_yolov2::DetectionArray
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:size __size) 0)
    ((:data __data) (let (r) (dotimes (i 0) (push (instance object_detection_yolov2::Detection :init) r)) r))
    )
   (send-super :init)
   (setq _header __header)
   (setq _size (round __size))
   (setq _data __data)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:size
   (&optional __size)
   (if __size (setq _size __size)) _size)
  (:data
   (&rest __data)
   (if (keywordp (car __data))
       (send* _data __data)
     (progn
       (if __data (setq _data (car __data)))
       _data)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; uint32 _size
    4
    ;; object_detection_yolov2/Detection[] _data
    (apply #'+ (send-all _data :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; uint32 _size
       (write-long _size s)
     ;; object_detection_yolov2/Detection[] _data
     (write-long (length _data) s)
     (dolist (elem _data)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; uint32 _size
     (setq _size (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; object_detection_yolov2/Detection[] _data
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _data (let (r) (dotimes (i n) (push (instance object_detection_yolov2::Detection :init) r)) r))
     (dolist (elem- _data)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get object_detection_yolov2::DetectionArray :md5sum-) "f6f31e50896912f32e444660e6057cbd")
(setf (get object_detection_yolov2::DetectionArray :datatype-) "object_detection_yolov2/DetectionArray")
(setf (get object_detection_yolov2::DetectionArray :definition-)
      "Header header

# The size of the array
uint32 size
# The array containing all the detections
Detection[] data

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

================================================================================
MSG: object_detection_yolov2/Detection
Header header

string object_class
float32 p

uint16 x
uint16 y

float32 cam_x
float32 cam_y
float32 cam_z

uint16 width
uint16 height

")



(provide :object_detection_yolov2/DetectionArray "f6f31e50896912f32e444660e6057cbd")


