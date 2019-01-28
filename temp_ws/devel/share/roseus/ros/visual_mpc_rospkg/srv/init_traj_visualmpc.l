;; Auto-generated. Do not edit!


(when (boundp 'visual_mpc_rospkg::init_traj_visualmpc)
  (if (not (find-package "VISUAL_MPC_ROSPKG"))
    (make-package "VISUAL_MPC_ROSPKG"))
  (shadow 'init_traj_visualmpc (find-package "VISUAL_MPC_ROSPKG")))
(unless (find-package "VISUAL_MPC_ROSPKG::INIT_TRAJ_VISUALMPC")
  (make-package "VISUAL_MPC_ROSPKG::INIT_TRAJ_VISUALMPC"))
(unless (find-package "VISUAL_MPC_ROSPKG::INIT_TRAJ_VISUALMPCREQUEST")
  (make-package "VISUAL_MPC_ROSPKG::INIT_TRAJ_VISUALMPCREQUEST"))
(unless (find-package "VISUAL_MPC_ROSPKG::INIT_TRAJ_VISUALMPCRESPONSE")
  (make-package "VISUAL_MPC_ROSPKG::INIT_TRAJ_VISUALMPCRESPONSE"))

(in-package "ROS")

(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))




(defclass visual_mpc_rospkg::init_traj_visualmpcRequest
  :super ros::object
  :slots (_itr _igrp _goalmain _goalaux1 _save_subdir ))

(defmethod visual_mpc_rospkg::init_traj_visualmpcRequest
  (:init
   (&key
    ((:itr __itr) 0)
    ((:igrp __igrp) 0)
    ((:goalmain __goalmain) (instance sensor_msgs::Image :init))
    ((:goalaux1 __goalaux1) (instance sensor_msgs::Image :init))
    ((:save_subdir __save_subdir) "")
    )
   (send-super :init)
   (setq _itr (round __itr))
   (setq _igrp (round __igrp))
   (setq _goalmain __goalmain)
   (setq _goalaux1 __goalaux1)
   (setq _save_subdir (string __save_subdir))
   self)
  (:itr
   (&optional __itr)
   (if __itr (setq _itr __itr)) _itr)
  (:igrp
   (&optional __igrp)
   (if __igrp (setq _igrp __igrp)) _igrp)
  (:goalmain
   (&rest __goalmain)
   (if (keywordp (car __goalmain))
       (send* _goalmain __goalmain)
     (progn
       (if __goalmain (setq _goalmain (car __goalmain)))
       _goalmain)))
  (:goalaux1
   (&rest __goalaux1)
   (if (keywordp (car __goalaux1))
       (send* _goalaux1 __goalaux1)
     (progn
       (if __goalaux1 (setq _goalaux1 (car __goalaux1)))
       _goalaux1)))
  (:save_subdir
   (&optional __save_subdir)
   (if __save_subdir (setq _save_subdir __save_subdir)) _save_subdir)
  (:serialization-length
   ()
   (+
    ;; int64 _itr
    8
    ;; int64 _igrp
    8
    ;; sensor_msgs/Image _goalmain
    (send _goalmain :serialization-length)
    ;; sensor_msgs/Image _goalaux1
    (send _goalaux1 :serialization-length)
    ;; string _save_subdir
    4 (length _save_subdir)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int64 _itr
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _itr (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _itr) (= (length (_itr . bv)) 2)) ;; bignum
              (write-long (ash (elt (_itr . bv) 0) 0) s)
              (write-long (ash (elt (_itr . bv) 1) -1) s))
             ((and (class _itr) (= (length (_itr . bv)) 1)) ;; big1
              (write-long (elt (_itr . bv) 0) s)
              (write-long (if (>= _itr 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _itr s)(write-long (if (>= _itr 0) 0 #xffffffff) s)))
     ;; int64 _igrp
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _igrp (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _igrp) (= (length (_igrp . bv)) 2)) ;; bignum
              (write-long (ash (elt (_igrp . bv) 0) 0) s)
              (write-long (ash (elt (_igrp . bv) 1) -1) s))
             ((and (class _igrp) (= (length (_igrp . bv)) 1)) ;; big1
              (write-long (elt (_igrp . bv) 0) s)
              (write-long (if (>= _igrp 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _igrp s)(write-long (if (>= _igrp 0) 0 #xffffffff) s)))
     ;; sensor_msgs/Image _goalmain
       (send _goalmain :serialize s)
     ;; sensor_msgs/Image _goalaux1
       (send _goalaux1 :serialize s)
     ;; string _save_subdir
       (write-long (length _save_subdir) s) (princ _save_subdir s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int64 _itr
#+(or :alpha :irix6 :x86_64)
      (setf _itr (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _itr (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;; int64 _igrp
#+(or :alpha :irix6 :x86_64)
      (setf _igrp (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _igrp (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;; sensor_msgs/Image _goalmain
     (send _goalmain :deserialize buf ptr-) (incf ptr- (send _goalmain :serialization-length))
   ;; sensor_msgs/Image _goalaux1
     (send _goalaux1 :deserialize buf ptr-) (incf ptr- (send _goalaux1 :serialization-length))
   ;; string _save_subdir
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _save_subdir (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass visual_mpc_rospkg::init_traj_visualmpcResponse
  :super ros::object
  :slots ())

(defmethod visual_mpc_rospkg::init_traj_visualmpcResponse
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

(defclass visual_mpc_rospkg::init_traj_visualmpc
  :super ros::object
  :slots ())

(setf (get visual_mpc_rospkg::init_traj_visualmpc :md5sum-) "212549b9c1a4ea535ff2ca3d14d779c5")
(setf (get visual_mpc_rospkg::init_traj_visualmpc :datatype-) "visual_mpc_rospkg/init_traj_visualmpc")
(setf (get visual_mpc_rospkg::init_traj_visualmpc :request) visual_mpc_rospkg::init_traj_visualmpcRequest)
(setf (get visual_mpc_rospkg::init_traj_visualmpc :response) visual_mpc_rospkg::init_traj_visualmpcResponse)

(defmethod visual_mpc_rospkg::init_traj_visualmpcRequest
  (:response () (instance visual_mpc_rospkg::init_traj_visualmpcResponse :init)))

(setf (get visual_mpc_rospkg::init_traj_visualmpcRequest :md5sum-) "212549b9c1a4ea535ff2ca3d14d779c5")
(setf (get visual_mpc_rospkg::init_traj_visualmpcRequest :datatype-) "visual_mpc_rospkg/init_traj_visualmpcRequest")
(setf (get visual_mpc_rospkg::init_traj_visualmpcRequest :definition-)
      "int64 itr
int64 igrp
sensor_msgs/Image goalmain
sensor_msgs/Image goalaux1
string save_subdir

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
")

(setf (get visual_mpc_rospkg::init_traj_visualmpcResponse :md5sum-) "212549b9c1a4ea535ff2ca3d14d779c5")
(setf (get visual_mpc_rospkg::init_traj_visualmpcResponse :datatype-) "visual_mpc_rospkg/init_traj_visualmpcResponse")
(setf (get visual_mpc_rospkg::init_traj_visualmpcResponse :definition-)
      "int64 itr
int64 igrp
sensor_msgs/Image goalmain
sensor_msgs/Image goalaux1
string save_subdir

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
")



(provide :visual_mpc_rospkg/init_traj_visualmpc "212549b9c1a4ea535ff2ca3d14d779c5")


