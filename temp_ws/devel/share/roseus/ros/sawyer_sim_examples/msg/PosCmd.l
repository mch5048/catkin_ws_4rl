;; Auto-generated. Do not edit!


(when (boundp 'sawyer_sim_examples::PosCmd)
  (if (not (find-package "SAWYER_SIM_EXAMPLES"))
    (make-package "SAWYER_SIM_EXAMPLES"))
  (shadow 'PosCmd (find-package "SAWYER_SIM_EXAMPLES")))
(unless (find-package "SAWYER_SIM_EXAMPLES::POSCMD")
  (make-package "SAWYER_SIM_EXAMPLES::POSCMD"))

(in-package "ROS")
;;//! \htmlinclude PosCmd.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass sawyer_sim_examples::PosCmd
  :super ros::object
  :slots (_header _goal_cart_pos ))

(defmethod sawyer_sim_examples::PosCmd
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:goal_cart_pos __goal_cart_pos) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _header __header)
   (setq _goal_cart_pos __goal_cart_pos)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:goal_cart_pos
   (&optional __goal_cart_pos)
   (if __goal_cart_pos (setq _goal_cart_pos __goal_cart_pos)) _goal_cart_pos)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float64[] _goal_cart_pos
    (* 8    (length _goal_cart_pos)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float64[] _goal_cart_pos
     (write-long (length _goal_cart_pos) s)
     (dotimes (i (length _goal_cart_pos))
       (sys::poke (elt _goal_cart_pos i) (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float64[] _goal_cart_pos
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _goal_cart_pos (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _goal_cart_pos i) (sys::peek buf ptr- :double)) (incf ptr- 8)
     ))
   ;;
   self)
  )

(setf (get sawyer_sim_examples::PosCmd :md5sum-) "2a7aed1f0b63cf2c59e8c977d20aef5b")
(setf (get sawyer_sim_examples::PosCmd :datatype-) "sawyer_sim_examples/PosCmd")
(setf (get sawyer_sim_examples::PosCmd :definition-)
      "##the name of the output
Header header

float64[] goal_cart_pos

##the value to set output 


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



(provide :sawyer_sim_examples/PosCmd "2a7aed1f0b63cf2c59e8c977d20aef5b")


