;; Auto-generated. Do not edit!


(when (boundp 'gazebo_ext_msgs::SetSurfaceParams)
  (if (not (find-package "GAZEBO_EXT_MSGS"))
    (make-package "GAZEBO_EXT_MSGS"))
  (shadow 'SetSurfaceParams (find-package "GAZEBO_EXT_MSGS")))
(unless (find-package "GAZEBO_EXT_MSGS::SETSURFACEPARAMS")
  (make-package "GAZEBO_EXT_MSGS::SETSURFACEPARAMS"))
(unless (find-package "GAZEBO_EXT_MSGS::SETSURFACEPARAMSREQUEST")
  (make-package "GAZEBO_EXT_MSGS::SETSURFACEPARAMSREQUEST"))
(unless (find-package "GAZEBO_EXT_MSGS::SETSURFACEPARAMSRESPONSE")
  (make-package "GAZEBO_EXT_MSGS::SETSURFACEPARAMSRESPONSE"))

(in-package "ROS")





(defclass gazebo_ext_msgs::SetSurfaceParamsRequest
  :super ros::object
  :slots (_link_collision_name _elastic_modulus _mu1 _mu2 _mu_torsion _patch_radius _poisson_ratio ))

(defmethod gazebo_ext_msgs::SetSurfaceParamsRequest
  (:init
   (&key
    ((:link_collision_name __link_collision_name) "")
    ((:elastic_modulus __elastic_modulus) 0.0)
    ((:mu1 __mu1) 0.0)
    ((:mu2 __mu2) 0.0)
    ((:mu_torsion __mu_torsion) 0.0)
    ((:patch_radius __patch_radius) 0.0)
    ((:poisson_ratio __poisson_ratio) 0.0)
    )
   (send-super :init)
   (setq _link_collision_name (string __link_collision_name))
   (setq _elastic_modulus (float __elastic_modulus))
   (setq _mu1 (float __mu1))
   (setq _mu2 (float __mu2))
   (setq _mu_torsion (float __mu_torsion))
   (setq _patch_radius (float __patch_radius))
   (setq _poisson_ratio (float __poisson_ratio))
   self)
  (:link_collision_name
   (&optional __link_collision_name)
   (if __link_collision_name (setq _link_collision_name __link_collision_name)) _link_collision_name)
  (:elastic_modulus
   (&optional __elastic_modulus)
   (if __elastic_modulus (setq _elastic_modulus __elastic_modulus)) _elastic_modulus)
  (:mu1
   (&optional __mu1)
   (if __mu1 (setq _mu1 __mu1)) _mu1)
  (:mu2
   (&optional __mu2)
   (if __mu2 (setq _mu2 __mu2)) _mu2)
  (:mu_torsion
   (&optional __mu_torsion)
   (if __mu_torsion (setq _mu_torsion __mu_torsion)) _mu_torsion)
  (:patch_radius
   (&optional __patch_radius)
   (if __patch_radius (setq _patch_radius __patch_radius)) _patch_radius)
  (:poisson_ratio
   (&optional __poisson_ratio)
   (if __poisson_ratio (setq _poisson_ratio __poisson_ratio)) _poisson_ratio)
  (:serialization-length
   ()
   (+
    ;; string _link_collision_name
    4 (length _link_collision_name)
    ;; float64 _elastic_modulus
    8
    ;; float64 _mu1
    8
    ;; float64 _mu2
    8
    ;; float64 _mu_torsion
    8
    ;; float64 _patch_radius
    8
    ;; float64 _poisson_ratio
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _link_collision_name
       (write-long (length _link_collision_name) s) (princ _link_collision_name s)
     ;; float64 _elastic_modulus
       (sys::poke _elastic_modulus (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _mu1
       (sys::poke _mu1 (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _mu2
       (sys::poke _mu2 (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _mu_torsion
       (sys::poke _mu_torsion (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _patch_radius
       (sys::poke _patch_radius (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _poisson_ratio
       (sys::poke _poisson_ratio (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _link_collision_name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _link_collision_name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float64 _elastic_modulus
     (setq _elastic_modulus (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _mu1
     (setq _mu1 (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _mu2
     (setq _mu2 (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _mu_torsion
     (setq _mu_torsion (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _patch_radius
     (setq _patch_radius (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _poisson_ratio
     (setq _poisson_ratio (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(defclass gazebo_ext_msgs::SetSurfaceParamsResponse
  :super ros::object
  :slots (_success _status_message ))

(defmethod gazebo_ext_msgs::SetSurfaceParamsResponse
  (:init
   (&key
    ((:success __success) nil)
    ((:status_message __status_message) "")
    )
   (send-super :init)
   (setq _success __success)
   (setq _status_message (string __status_message))
   self)
  (:success
   (&optional __success)
   (if __success (setq _success __success)) _success)
  (:status_message
   (&optional __status_message)
   (if __status_message (setq _status_message __status_message)) _status_message)
  (:serialization-length
   ()
   (+
    ;; bool _success
    1
    ;; string _status_message
    4 (length _status_message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;; string _status_message
       (write-long (length _status_message) s) (princ _status_message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string _status_message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _status_message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass gazebo_ext_msgs::SetSurfaceParams
  :super ros::object
  :slots ())

(setf (get gazebo_ext_msgs::SetSurfaceParams :md5sum-) "5b42f161dd595c17f404172b095c2d93")
(setf (get gazebo_ext_msgs::SetSurfaceParams :datatype-) "gazebo_ext_msgs/SetSurfaceParams")
(setf (get gazebo_ext_msgs::SetSurfaceParams :request) gazebo_ext_msgs::SetSurfaceParamsRequest)
(setf (get gazebo_ext_msgs::SetSurfaceParams :response) gazebo_ext_msgs::SetSurfaceParamsResponse)

(defmethod gazebo_ext_msgs::SetSurfaceParamsRequest
  (:response () (instance gazebo_ext_msgs::SetSurfaceParamsResponse :init)))

(setf (get gazebo_ext_msgs::SetSurfaceParamsRequest :md5sum-) "5b42f161dd595c17f404172b095c2d93")
(setf (get gazebo_ext_msgs::SetSurfaceParamsRequest :datatype-) "gazebo_ext_msgs/SetSurfaceParamsRequest")
(setf (get gazebo_ext_msgs::SetSurfaceParamsRequest :definition-)
      "string link_collision_name
float64 elastic_modulus
float64 mu1
float64 mu2
float64 mu_torsion
float64 patch_radius
float64 poisson_ratio
---
bool success
string status_message
")

(setf (get gazebo_ext_msgs::SetSurfaceParamsResponse :md5sum-) "5b42f161dd595c17f404172b095c2d93")
(setf (get gazebo_ext_msgs::SetSurfaceParamsResponse :datatype-) "gazebo_ext_msgs/SetSurfaceParamsResponse")
(setf (get gazebo_ext_msgs::SetSurfaceParamsResponse :definition-)
      "string link_collision_name
float64 elastic_modulus
float64 mu1
float64 mu2
float64 mu_torsion
float64 patch_radius
float64 poisson_ratio
---
bool success
string status_message
")



(provide :gazebo_ext_msgs/SetSurfaceParams "5b42f161dd595c17f404172b095c2d93")


