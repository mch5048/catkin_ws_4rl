;; Auto-generated. Do not edit!


(when (boundp 'gazebo_ext_msgs::GetCollisionNames)
  (if (not (find-package "GAZEBO_EXT_MSGS"))
    (make-package "GAZEBO_EXT_MSGS"))
  (shadow 'GetCollisionNames (find-package "GAZEBO_EXT_MSGS")))
(unless (find-package "GAZEBO_EXT_MSGS::GETCOLLISIONNAMES")
  (make-package "GAZEBO_EXT_MSGS::GETCOLLISIONNAMES"))
(unless (find-package "GAZEBO_EXT_MSGS::GETCOLLISIONNAMESREQUEST")
  (make-package "GAZEBO_EXT_MSGS::GETCOLLISIONNAMESREQUEST"))
(unless (find-package "GAZEBO_EXT_MSGS::GETCOLLISIONNAMESRESPONSE")
  (make-package "GAZEBO_EXT_MSGS::GETCOLLISIONNAMESRESPONSE"))

(in-package "ROS")





(defclass gazebo_ext_msgs::GetCollisionNamesRequest
  :super ros::object
  :slots (_link_names ))

(defmethod gazebo_ext_msgs::GetCollisionNamesRequest
  (:init
   (&key
    ((:link_names __link_names) (let (r) (dotimes (i 0) (push "" r)) r))
    )
   (send-super :init)
   (setq _link_names __link_names)
   self)
  (:link_names
   (&optional __link_names)
   (if __link_names (setq _link_names __link_names)) _link_names)
  (:serialization-length
   ()
   (+
    ;; string[] _link_names
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _link_names)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string[] _link_names
     (write-long (length _link_names) s)
     (dolist (elem _link_names)
       (write-long (length elem) s) (princ elem s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string[] _link_names
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _link_names (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _link_names i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;;
   self)
  )

(defclass gazebo_ext_msgs::GetCollisionNamesResponse
  :super ros::object
  :slots (_link_collision_names _success _status_message ))

(defmethod gazebo_ext_msgs::GetCollisionNamesResponse
  (:init
   (&key
    ((:link_collision_names __link_collision_names) (let (r) (dotimes (i 0) (push "" r)) r))
    ((:success __success) nil)
    ((:status_message __status_message) "")
    )
   (send-super :init)
   (setq _link_collision_names __link_collision_names)
   (setq _success __success)
   (setq _status_message (string __status_message))
   self)
  (:link_collision_names
   (&optional __link_collision_names)
   (if __link_collision_names (setq _link_collision_names __link_collision_names)) _link_collision_names)
  (:success
   (&optional __success)
   (if __success (setq _success __success)) _success)
  (:status_message
   (&optional __status_message)
   (if __status_message (setq _status_message __status_message)) _status_message)
  (:serialization-length
   ()
   (+
    ;; string[] _link_collision_names
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _link_collision_names)) 4
    ;; bool _success
    1
    ;; string _status_message
    4 (length _status_message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string[] _link_collision_names
     (write-long (length _link_collision_names) s)
     (dolist (elem _link_collision_names)
       (write-long (length elem) s) (princ elem s)
       )
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;; string _status_message
       (write-long (length _status_message) s) (princ _status_message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string[] _link_collision_names
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _link_collision_names (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _link_collision_names i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string _status_message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _status_message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass gazebo_ext_msgs::GetCollisionNames
  :super ros::object
  :slots ())

(setf (get gazebo_ext_msgs::GetCollisionNames :md5sum-) "440bd39ad26bc774c2ef6bcfe06d56ea")
(setf (get gazebo_ext_msgs::GetCollisionNames :datatype-) "gazebo_ext_msgs/GetCollisionNames")
(setf (get gazebo_ext_msgs::GetCollisionNames :request) gazebo_ext_msgs::GetCollisionNamesRequest)
(setf (get gazebo_ext_msgs::GetCollisionNames :response) gazebo_ext_msgs::GetCollisionNamesResponse)

(defmethod gazebo_ext_msgs::GetCollisionNamesRequest
  (:response () (instance gazebo_ext_msgs::GetCollisionNamesResponse :init)))

(setf (get gazebo_ext_msgs::GetCollisionNamesRequest :md5sum-) "440bd39ad26bc774c2ef6bcfe06d56ea")
(setf (get gazebo_ext_msgs::GetCollisionNamesRequest :datatype-) "gazebo_ext_msgs/GetCollisionNamesRequest")
(setf (get gazebo_ext_msgs::GetCollisionNamesRequest :definition-)
      "string[] link_names
---
string[] link_collision_names
bool success
string status_message
")

(setf (get gazebo_ext_msgs::GetCollisionNamesResponse :md5sum-) "440bd39ad26bc774c2ef6bcfe06d56ea")
(setf (get gazebo_ext_msgs::GetCollisionNamesResponse :datatype-) "gazebo_ext_msgs/GetCollisionNamesResponse")
(setf (get gazebo_ext_msgs::GetCollisionNamesResponse :definition-)
      "string[] link_names
---
string[] link_collision_names
bool success
string status_message
")



(provide :gazebo_ext_msgs/GetCollisionNames "440bd39ad26bc774c2ef6bcfe06d56ea")


