;; Auto-generated. Do not edit!


(when (boundp 'gazebo_ext_msgs::GetVisualNames)
  (if (not (find-package "GAZEBO_EXT_MSGS"))
    (make-package "GAZEBO_EXT_MSGS"))
  (shadow 'GetVisualNames (find-package "GAZEBO_EXT_MSGS")))
(unless (find-package "GAZEBO_EXT_MSGS::GETVISUALNAMES")
  (make-package "GAZEBO_EXT_MSGS::GETVISUALNAMES"))
(unless (find-package "GAZEBO_EXT_MSGS::GETVISUALNAMESREQUEST")
  (make-package "GAZEBO_EXT_MSGS::GETVISUALNAMESREQUEST"))
(unless (find-package "GAZEBO_EXT_MSGS::GETVISUALNAMESRESPONSE")
  (make-package "GAZEBO_EXT_MSGS::GETVISUALNAMESRESPONSE"))

(in-package "ROS")





(defclass gazebo_ext_msgs::GetVisualNamesRequest
  :super ros::object
  :slots (_link_names ))

(defmethod gazebo_ext_msgs::GetVisualNamesRequest
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

(defclass gazebo_ext_msgs::GetVisualNamesResponse
  :super ros::object
  :slots (_link_visual_names _success _status_message ))

(defmethod gazebo_ext_msgs::GetVisualNamesResponse
  (:init
   (&key
    ((:link_visual_names __link_visual_names) (let (r) (dotimes (i 0) (push "" r)) r))
    ((:success __success) nil)
    ((:status_message __status_message) "")
    )
   (send-super :init)
   (setq _link_visual_names __link_visual_names)
   (setq _success __success)
   (setq _status_message (string __status_message))
   self)
  (:link_visual_names
   (&optional __link_visual_names)
   (if __link_visual_names (setq _link_visual_names __link_visual_names)) _link_visual_names)
  (:success
   (&optional __success)
   (if __success (setq _success __success)) _success)
  (:status_message
   (&optional __status_message)
   (if __status_message (setq _status_message __status_message)) _status_message)
  (:serialization-length
   ()
   (+
    ;; string[] _link_visual_names
    (apply #'+ (mapcar #'(lambda (x) (+ 4 (length x))) _link_visual_names)) 4
    ;; bool _success
    1
    ;; string _status_message
    4 (length _status_message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string[] _link_visual_names
     (write-long (length _link_visual_names) s)
     (dolist (elem _link_visual_names)
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
   ;; string[] _link_visual_names
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _link_visual_names (make-list n))
     (dotimes (i n)
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setf (elt _link_visual_names i) (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
     ))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string _status_message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _status_message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass gazebo_ext_msgs::GetVisualNames
  :super ros::object
  :slots ())

(setf (get gazebo_ext_msgs::GetVisualNames :md5sum-) "84bee8e2f694c3877c28343b65184d47")
(setf (get gazebo_ext_msgs::GetVisualNames :datatype-) "gazebo_ext_msgs/GetVisualNames")
(setf (get gazebo_ext_msgs::GetVisualNames :request) gazebo_ext_msgs::GetVisualNamesRequest)
(setf (get gazebo_ext_msgs::GetVisualNames :response) gazebo_ext_msgs::GetVisualNamesResponse)

(defmethod gazebo_ext_msgs::GetVisualNamesRequest
  (:response () (instance gazebo_ext_msgs::GetVisualNamesResponse :init)))

(setf (get gazebo_ext_msgs::GetVisualNamesRequest :md5sum-) "84bee8e2f694c3877c28343b65184d47")
(setf (get gazebo_ext_msgs::GetVisualNamesRequest :datatype-) "gazebo_ext_msgs/GetVisualNamesRequest")
(setf (get gazebo_ext_msgs::GetVisualNamesRequest :definition-)
      "string[] link_names
---
string[] link_visual_names
bool success
string status_message
")

(setf (get gazebo_ext_msgs::GetVisualNamesResponse :md5sum-) "84bee8e2f694c3877c28343b65184d47")
(setf (get gazebo_ext_msgs::GetVisualNamesResponse :datatype-) "gazebo_ext_msgs/GetVisualNamesResponse")
(setf (get gazebo_ext_msgs::GetVisualNamesResponse :definition-)
      "string[] link_names
---
string[] link_visual_names
bool success
string status_message
")



(provide :gazebo_ext_msgs/GetVisualNames "84bee8e2f694c3877c28343b65184d47")


