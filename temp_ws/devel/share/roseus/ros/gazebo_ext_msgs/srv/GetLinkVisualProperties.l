;; Auto-generated. Do not edit!


(when (boundp 'gazebo_ext_msgs::GetLinkVisualProperties)
  (if (not (find-package "GAZEBO_EXT_MSGS"))
    (make-package "GAZEBO_EXT_MSGS"))
  (shadow 'GetLinkVisualProperties (find-package "GAZEBO_EXT_MSGS")))
(unless (find-package "GAZEBO_EXT_MSGS::GETLINKVISUALPROPERTIES")
  (make-package "GAZEBO_EXT_MSGS::GETLINKVISUALPROPERTIES"))
(unless (find-package "GAZEBO_EXT_MSGS::GETLINKVISUALPROPERTIESREQUEST")
  (make-package "GAZEBO_EXT_MSGS::GETLINKVISUALPROPERTIESREQUEST"))
(unless (find-package "GAZEBO_EXT_MSGS::GETLINKVISUALPROPERTIESRESPONSE")
  (make-package "GAZEBO_EXT_MSGS::GETLINKVISUALPROPERTIESRESPONSE"))

(in-package "ROS")



(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass gazebo_ext_msgs::GetLinkVisualPropertiesRequest
  :super ros::object
  :slots (_link_visual_name ))

(defmethod gazebo_ext_msgs::GetLinkVisualPropertiesRequest
  (:init
   (&key
    ((:link_visual_name __link_visual_name) "")
    )
   (send-super :init)
   (setq _link_visual_name (string __link_visual_name))
   self)
  (:link_visual_name
   (&optional __link_visual_name)
   (if __link_visual_name (setq _link_visual_name __link_visual_name)) _link_visual_name)
  (:serialization-length
   ()
   (+
    ;; string _link_visual_name
    4 (length _link_visual_name)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _link_visual_name
       (write-long (length _link_visual_name) s) (princ _link_visual_name s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _link_visual_name
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _link_visual_name (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass gazebo_ext_msgs::GetLinkVisualPropertiesResponse
  :super ros::object
  :slots (_ambient _diffuse _specular _emissive _success _status_message ))

(defmethod gazebo_ext_msgs::GetLinkVisualPropertiesResponse
  (:init
   (&key
    ((:ambient __ambient) (instance std_msgs::ColorRGBA :init))
    ((:diffuse __diffuse) (instance std_msgs::ColorRGBA :init))
    ((:specular __specular) (instance std_msgs::ColorRGBA :init))
    ((:emissive __emissive) (instance std_msgs::ColorRGBA :init))
    ((:success __success) nil)
    ((:status_message __status_message) "")
    )
   (send-super :init)
   (setq _ambient __ambient)
   (setq _diffuse __diffuse)
   (setq _specular __specular)
   (setq _emissive __emissive)
   (setq _success __success)
   (setq _status_message (string __status_message))
   self)
  (:ambient
   (&rest __ambient)
   (if (keywordp (car __ambient))
       (send* _ambient __ambient)
     (progn
       (if __ambient (setq _ambient (car __ambient)))
       _ambient)))
  (:diffuse
   (&rest __diffuse)
   (if (keywordp (car __diffuse))
       (send* _diffuse __diffuse)
     (progn
       (if __diffuse (setq _diffuse (car __diffuse)))
       _diffuse)))
  (:specular
   (&rest __specular)
   (if (keywordp (car __specular))
       (send* _specular __specular)
     (progn
       (if __specular (setq _specular (car __specular)))
       _specular)))
  (:emissive
   (&rest __emissive)
   (if (keywordp (car __emissive))
       (send* _emissive __emissive)
     (progn
       (if __emissive (setq _emissive (car __emissive)))
       _emissive)))
  (:success
   (&optional __success)
   (if __success (setq _success __success)) _success)
  (:status_message
   (&optional __status_message)
   (if __status_message (setq _status_message __status_message)) _status_message)
  (:serialization-length
   ()
   (+
    ;; std_msgs/ColorRGBA _ambient
    (send _ambient :serialization-length)
    ;; std_msgs/ColorRGBA _diffuse
    (send _diffuse :serialization-length)
    ;; std_msgs/ColorRGBA _specular
    (send _specular :serialization-length)
    ;; std_msgs/ColorRGBA _emissive
    (send _emissive :serialization-length)
    ;; bool _success
    1
    ;; string _status_message
    4 (length _status_message)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/ColorRGBA _ambient
       (send _ambient :serialize s)
     ;; std_msgs/ColorRGBA _diffuse
       (send _diffuse :serialize s)
     ;; std_msgs/ColorRGBA _specular
       (send _specular :serialize s)
     ;; std_msgs/ColorRGBA _emissive
       (send _emissive :serialize s)
     ;; bool _success
       (if _success (write-byte -1 s) (write-byte 0 s))
     ;; string _status_message
       (write-long (length _status_message) s) (princ _status_message s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/ColorRGBA _ambient
     (send _ambient :deserialize buf ptr-) (incf ptr- (send _ambient :serialization-length))
   ;; std_msgs/ColorRGBA _diffuse
     (send _diffuse :deserialize buf ptr-) (incf ptr- (send _diffuse :serialization-length))
   ;; std_msgs/ColorRGBA _specular
     (send _specular :deserialize buf ptr-) (incf ptr- (send _specular :serialization-length))
   ;; std_msgs/ColorRGBA _emissive
     (send _emissive :deserialize buf ptr-) (incf ptr- (send _emissive :serialization-length))
   ;; bool _success
     (setq _success (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string _status_message
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _status_message (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass gazebo_ext_msgs::GetLinkVisualProperties
  :super ros::object
  :slots ())

(setf (get gazebo_ext_msgs::GetLinkVisualProperties :md5sum-) "565eef77cf4ad97635bdc8bf4af90f87")
(setf (get gazebo_ext_msgs::GetLinkVisualProperties :datatype-) "gazebo_ext_msgs/GetLinkVisualProperties")
(setf (get gazebo_ext_msgs::GetLinkVisualProperties :request) gazebo_ext_msgs::GetLinkVisualPropertiesRequest)
(setf (get gazebo_ext_msgs::GetLinkVisualProperties :response) gazebo_ext_msgs::GetLinkVisualPropertiesResponse)

(defmethod gazebo_ext_msgs::GetLinkVisualPropertiesRequest
  (:response () (instance gazebo_ext_msgs::GetLinkVisualPropertiesResponse :init)))

(setf (get gazebo_ext_msgs::GetLinkVisualPropertiesRequest :md5sum-) "565eef77cf4ad97635bdc8bf4af90f87")
(setf (get gazebo_ext_msgs::GetLinkVisualPropertiesRequest :datatype-) "gazebo_ext_msgs/GetLinkVisualPropertiesRequest")
(setf (get gazebo_ext_msgs::GetLinkVisualPropertiesRequest :definition-)
      "string link_visual_name
---
std_msgs/ColorRGBA ambient
std_msgs/ColorRGBA diffuse
std_msgs/ColorRGBA specular
std_msgs/ColorRGBA emissive
bool success
string status_message

================================================================================
MSG: std_msgs/ColorRGBA
float32 r
float32 g
float32 b
float32 a
")

(setf (get gazebo_ext_msgs::GetLinkVisualPropertiesResponse :md5sum-) "565eef77cf4ad97635bdc8bf4af90f87")
(setf (get gazebo_ext_msgs::GetLinkVisualPropertiesResponse :datatype-) "gazebo_ext_msgs/GetLinkVisualPropertiesResponse")
(setf (get gazebo_ext_msgs::GetLinkVisualPropertiesResponse :definition-)
      "string link_visual_name
---
std_msgs/ColorRGBA ambient
std_msgs/ColorRGBA diffuse
std_msgs/ColorRGBA specular
std_msgs/ColorRGBA emissive
bool success
string status_message

================================================================================
MSG: std_msgs/ColorRGBA
float32 r
float32 g
float32 b
float32 a
")



(provide :gazebo_ext_msgs/GetLinkVisualProperties "565eef77cf4ad97635bdc8bf4af90f87")


