;; Auto-generated. Do not edit!


(when (boundp 'gazebo_ext_msgs::SetSkyProperties)
  (if (not (find-package "GAZEBO_EXT_MSGS"))
    (make-package "GAZEBO_EXT_MSGS"))
  (shadow 'SetSkyProperties (find-package "GAZEBO_EXT_MSGS")))
(unless (find-package "GAZEBO_EXT_MSGS::SETSKYPROPERTIES")
  (make-package "GAZEBO_EXT_MSGS::SETSKYPROPERTIES"))
(unless (find-package "GAZEBO_EXT_MSGS::SETSKYPROPERTIESREQUEST")
  (make-package "GAZEBO_EXT_MSGS::SETSKYPROPERTIESREQUEST"))
(unless (find-package "GAZEBO_EXT_MSGS::SETSKYPROPERTIESRESPONSE")
  (make-package "GAZEBO_EXT_MSGS::SETSKYPROPERTIESRESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))




(defclass gazebo_ext_msgs::SetSkyPropertiesRequest
  :super ros::object
  :slots (_time _sunrise _sunset _wind_speed _wind_direction _cloud_ambient _humidity _mean_cloud_size ))

(defmethod gazebo_ext_msgs::SetSkyPropertiesRequest
  (:init
   (&key
    ((:time __time) 0.0)
    ((:sunrise __sunrise) 0.0)
    ((:sunset __sunset) 0.0)
    ((:wind_speed __wind_speed) 0.0)
    ((:wind_direction __wind_direction) 0.0)
    ((:cloud_ambient __cloud_ambient) (instance std_msgs::ColorRGBA :init))
    ((:humidity __humidity) 0.0)
    ((:mean_cloud_size __mean_cloud_size) 0.0)
    )
   (send-super :init)
   (setq _time (float __time))
   (setq _sunrise (float __sunrise))
   (setq _sunset (float __sunset))
   (setq _wind_speed (float __wind_speed))
   (setq _wind_direction (float __wind_direction))
   (setq _cloud_ambient __cloud_ambient)
   (setq _humidity (float __humidity))
   (setq _mean_cloud_size (float __mean_cloud_size))
   self)
  (:time
   (&optional __time)
   (if __time (setq _time __time)) _time)
  (:sunrise
   (&optional __sunrise)
   (if __sunrise (setq _sunrise __sunrise)) _sunrise)
  (:sunset
   (&optional __sunset)
   (if __sunset (setq _sunset __sunset)) _sunset)
  (:wind_speed
   (&optional __wind_speed)
   (if __wind_speed (setq _wind_speed __wind_speed)) _wind_speed)
  (:wind_direction
   (&optional __wind_direction)
   (if __wind_direction (setq _wind_direction __wind_direction)) _wind_direction)
  (:cloud_ambient
   (&rest __cloud_ambient)
   (if (keywordp (car __cloud_ambient))
       (send* _cloud_ambient __cloud_ambient)
     (progn
       (if __cloud_ambient (setq _cloud_ambient (car __cloud_ambient)))
       _cloud_ambient)))
  (:humidity
   (&optional __humidity)
   (if __humidity (setq _humidity __humidity)) _humidity)
  (:mean_cloud_size
   (&optional __mean_cloud_size)
   (if __mean_cloud_size (setq _mean_cloud_size __mean_cloud_size)) _mean_cloud_size)
  (:serialization-length
   ()
   (+
    ;; float64 _time
    8
    ;; float64 _sunrise
    8
    ;; float64 _sunset
    8
    ;; float64 _wind_speed
    8
    ;; float64 _wind_direction
    8
    ;; std_msgs/ColorRGBA _cloud_ambient
    (send _cloud_ambient :serialization-length)
    ;; float64 _humidity
    8
    ;; float64 _mean_cloud_size
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float64 _time
       (sys::poke _time (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _sunrise
       (sys::poke _sunrise (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _sunset
       (sys::poke _sunset (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _wind_speed
       (sys::poke _wind_speed (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _wind_direction
       (sys::poke _wind_direction (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; std_msgs/ColorRGBA _cloud_ambient
       (send _cloud_ambient :serialize s)
     ;; float64 _humidity
       (sys::poke _humidity (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;; float64 _mean_cloud_size
       (sys::poke _mean_cloud_size (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float64 _time
     (setq _time (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _sunrise
     (setq _sunrise (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _sunset
     (setq _sunset (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _wind_speed
     (setq _wind_speed (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _wind_direction
     (setq _wind_direction (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; std_msgs/ColorRGBA _cloud_ambient
     (send _cloud_ambient :deserialize buf ptr-) (incf ptr- (send _cloud_ambient :serialization-length))
   ;; float64 _humidity
     (setq _humidity (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;; float64 _mean_cloud_size
     (setq _mean_cloud_size (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(defclass gazebo_ext_msgs::SetSkyPropertiesResponse
  :super ros::object
  :slots (_success _status_message ))

(defmethod gazebo_ext_msgs::SetSkyPropertiesResponse
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

(defclass gazebo_ext_msgs::SetSkyProperties
  :super ros::object
  :slots ())

(setf (get gazebo_ext_msgs::SetSkyProperties :md5sum-) "58ab80a272655f0012daa4ccd6a59539")
(setf (get gazebo_ext_msgs::SetSkyProperties :datatype-) "gazebo_ext_msgs/SetSkyProperties")
(setf (get gazebo_ext_msgs::SetSkyProperties :request) gazebo_ext_msgs::SetSkyPropertiesRequest)
(setf (get gazebo_ext_msgs::SetSkyProperties :response) gazebo_ext_msgs::SetSkyPropertiesResponse)

(defmethod gazebo_ext_msgs::SetSkyPropertiesRequest
  (:response () (instance gazebo_ext_msgs::SetSkyPropertiesResponse :init)))

(setf (get gazebo_ext_msgs::SetSkyPropertiesRequest :md5sum-) "58ab80a272655f0012daa4ccd6a59539")
(setf (get gazebo_ext_msgs::SetSkyPropertiesRequest :datatype-) "gazebo_ext_msgs/SetSkyPropertiesRequest")
(setf (get gazebo_ext_msgs::SetSkyPropertiesRequest :definition-)
      "float64 time
float64 sunrise
float64 sunset
float64 wind_speed
float64 wind_direction
std_msgs/ColorRGBA cloud_ambient
float64 humidity
float64 mean_cloud_size

================================================================================
MSG: std_msgs/ColorRGBA
float32 r
float32 g
float32 b
float32 a
---
bool success
string status_message

")

(setf (get gazebo_ext_msgs::SetSkyPropertiesResponse :md5sum-) "58ab80a272655f0012daa4ccd6a59539")
(setf (get gazebo_ext_msgs::SetSkyPropertiesResponse :datatype-) "gazebo_ext_msgs/SetSkyPropertiesResponse")
(setf (get gazebo_ext_msgs::SetSkyPropertiesResponse :definition-)
      "float64 time
float64 sunrise
float64 sunset
float64 wind_speed
float64 wind_direction
std_msgs/ColorRGBA cloud_ambient
float64 humidity
float64 mean_cloud_size

================================================================================
MSG: std_msgs/ColorRGBA
float32 r
float32 g
float32 b
float32 a
---
bool success
string status_message

")



(provide :gazebo_ext_msgs/SetSkyProperties "58ab80a272655f0012daa4ccd6a59539")


