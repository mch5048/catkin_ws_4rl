;; Auto-generated. Do not edit!


(when (boundp 'berkeley_sawyer::get_action)
  (if (not (find-package "BERKELEY_SAWYER"))
    (make-package "BERKELEY_SAWYER"))
  (shadow 'get_action (find-package "BERKELEY_SAWYER")))
(unless (find-package "BERKELEY_SAWYER::GET_ACTION")
  (make-package "BERKELEY_SAWYER::GET_ACTION"))
(unless (find-package "BERKELEY_SAWYER::GET_ACTIONREQUEST")
  (make-package "BERKELEY_SAWYER::GET_ACTIONREQUEST"))
(unless (find-package "BERKELEY_SAWYER::GET_ACTIONRESPONSE")
  (make-package "BERKELEY_SAWYER::GET_ACTIONRESPONSE"))

(in-package "ROS")

(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))




(defclass berkeley_sawyer::get_actionRequest
  :super ros::object
  :slots (_main _aux1 _state _desig_pos_aux1 _goal_pos_aux1 ))

(defmethod berkeley_sawyer::get_actionRequest
  (:init
   (&key
    ((:main __main) (instance sensor_msgs::Image :init))
    ((:aux1 __aux1) (instance sensor_msgs::Image :init))
    ((:state __state) (make-array 3 :initial-element 0.0 :element-type :float))
    ((:desig_pos_aux1 __desig_pos_aux1) (make-array 4 :initial-element 0 :element-type :integer))
    ((:goal_pos_aux1 __goal_pos_aux1) (make-array 4 :initial-element 0 :element-type :integer))
    )
   (send-super :init)
   (setq _main __main)
   (setq _aux1 __aux1)
   (setq _state __state)
   (setq _desig_pos_aux1 __desig_pos_aux1)
   (setq _goal_pos_aux1 __goal_pos_aux1)
   self)
  (:main
   (&rest __main)
   (if (keywordp (car __main))
       (send* _main __main)
     (progn
       (if __main (setq _main (car __main)))
       _main)))
  (:aux1
   (&rest __aux1)
   (if (keywordp (car __aux1))
       (send* _aux1 __aux1)
     (progn
       (if __aux1 (setq _aux1 (car __aux1)))
       _aux1)))
  (:state
   (&optional __state)
   (if __state (setq _state __state)) _state)
  (:desig_pos_aux1
   (&optional __desig_pos_aux1)
   (if __desig_pos_aux1 (setq _desig_pos_aux1 __desig_pos_aux1)) _desig_pos_aux1)
  (:goal_pos_aux1
   (&optional __goal_pos_aux1)
   (if __goal_pos_aux1 (setq _goal_pos_aux1 __goal_pos_aux1)) _goal_pos_aux1)
  (:serialization-length
   ()
   (+
    ;; sensor_msgs/Image _main
    (send _main :serialization-length)
    ;; sensor_msgs/Image _aux1
    (send _aux1 :serialization-length)
    ;; float32[3] _state
    (* 4    3)
    ;; int64[4] _desig_pos_aux1
    (* 8    4)
    ;; int64[4] _goal_pos_aux1
    (* 8    4)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; sensor_msgs/Image _main
       (send _main :serialize s)
     ;; sensor_msgs/Image _aux1
       (send _aux1 :serialize s)
     ;; float32[3] _state
     (dotimes (i 3)
       (sys::poke (elt _state i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; int64[4] _desig_pos_aux1
     (dotimes (i 4)
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke (elt _desig_pos_aux1 i) (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class (elt _desig_pos_aux1 i)) (= (length ((elt _desig_pos_aux1 i) . bv)) 2)) ;; bignum
              (write-long (ash (elt ((elt _desig_pos_aux1 i) . bv) 0) 0) s)
              (write-long (ash (elt ((elt _desig_pos_aux1 i) . bv) 1) -1) s))
             ((and (class (elt _desig_pos_aux1 i)) (= (length ((elt _desig_pos_aux1 i) . bv)) 1)) ;; big1
              (write-long (elt ((elt _desig_pos_aux1 i) . bv) 0) s)
              (write-long (if (>= (elt _desig_pos_aux1 i) 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long (elt _desig_pos_aux1 i) s)(write-long (if (>= (elt _desig_pos_aux1 i) 0) 0 #xffffffff) s)))
       )
     ;; int64[4] _goal_pos_aux1
     (dotimes (i 4)
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke (elt _goal_pos_aux1 i) (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class (elt _goal_pos_aux1 i)) (= (length ((elt _goal_pos_aux1 i) . bv)) 2)) ;; bignum
              (write-long (ash (elt ((elt _goal_pos_aux1 i) . bv) 0) 0) s)
              (write-long (ash (elt ((elt _goal_pos_aux1 i) . bv) 1) -1) s))
             ((and (class (elt _goal_pos_aux1 i)) (= (length ((elt _goal_pos_aux1 i) . bv)) 1)) ;; big1
              (write-long (elt ((elt _goal_pos_aux1 i) . bv) 0) s)
              (write-long (if (>= (elt _goal_pos_aux1 i) 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long (elt _goal_pos_aux1 i) s)(write-long (if (>= (elt _goal_pos_aux1 i) 0) 0 #xffffffff) s)))
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; sensor_msgs/Image _main
     (send _main :deserialize buf ptr-) (incf ptr- (send _main :serialization-length))
   ;; sensor_msgs/Image _aux1
     (send _aux1 :deserialize buf ptr-) (incf ptr- (send _aux1 :serialization-length))
   ;; float32[3] _state
   (dotimes (i (length _state))
     (setf (elt _state i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; int64[4] _desig_pos_aux1
   (dotimes (i (length _desig_pos_aux1))
#+(or :alpha :irix6 :x86_64)
      (setf (elt _desig_pos_aux1 i) (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf (elt _desig_pos_aux1 i) (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
     )
   ;; int64[4] _goal_pos_aux1
   (dotimes (i (length _goal_pos_aux1))
#+(or :alpha :irix6 :x86_64)
      (setf (elt _goal_pos_aux1 i) (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf (elt _goal_pos_aux1 i) (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
     )
   ;;
   self)
  )

(defclass berkeley_sawyer::get_actionResponse
  :super ros::object
  :slots (_action ))

(defmethod berkeley_sawyer::get_actionResponse
  (:init
   (&key
    ((:action __action) (make-array 4 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _action __action)
   self)
  (:action
   (&optional __action)
   (if __action (setq _action __action)) _action)
  (:serialization-length
   ()
   (+
    ;; float32[4] _action
    (* 4    4)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[4] _action
     (dotimes (i 4)
       (sys::poke (elt _action i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[4] _action
   (dotimes (i (length _action))
     (setf (elt _action i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;;
   self)
  )

(defclass berkeley_sawyer::get_action
  :super ros::object
  :slots ())

(setf (get berkeley_sawyer::get_action :md5sum-) "ed42212579cedad52c84913bafdfccf2")
(setf (get berkeley_sawyer::get_action :datatype-) "berkeley_sawyer/get_action")
(setf (get berkeley_sawyer::get_action :request) berkeley_sawyer::get_actionRequest)
(setf (get berkeley_sawyer::get_action :response) berkeley_sawyer::get_actionResponse)

(defmethod berkeley_sawyer::get_actionRequest
  (:response () (instance berkeley_sawyer::get_actionResponse :init)))

(setf (get berkeley_sawyer::get_actionRequest :md5sum-) "ed42212579cedad52c84913bafdfccf2")
(setf (get berkeley_sawyer::get_actionRequest :datatype-) "berkeley_sawyer/get_actionRequest")
(setf (get berkeley_sawyer::get_actionRequest :definition-)
      "sensor_msgs/Image main
sensor_msgs/Image aux1
float32[3] state
int64[4] desig_pos_aux1
int64[4] goal_pos_aux1

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
---
float32[4] action


")

(setf (get berkeley_sawyer::get_actionResponse :md5sum-) "ed42212579cedad52c84913bafdfccf2")
(setf (get berkeley_sawyer::get_actionResponse :datatype-) "berkeley_sawyer/get_actionResponse")
(setf (get berkeley_sawyer::get_actionResponse :definition-)
      "sensor_msgs/Image main
sensor_msgs/Image aux1
float32[3] state
int64[4] desig_pos_aux1
int64[4] goal_pos_aux1

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
---
float32[4] action


")



(provide :berkeley_sawyer/get_action "ed42212579cedad52c84913bafdfccf2")


