;; Auto-generated. Do not edit!


(when (boundp 'rospy_message_converter::Uint8ArrayTestMessage)
  (if (not (find-package "ROSPY_MESSAGE_CONVERTER"))
    (make-package "ROSPY_MESSAGE_CONVERTER"))
  (shadow 'Uint8ArrayTestMessage (find-package "ROSPY_MESSAGE_CONVERTER")))
(unless (find-package "ROSPY_MESSAGE_CONVERTER::UINT8ARRAYTESTMESSAGE")
  (make-package "ROSPY_MESSAGE_CONVERTER::UINT8ARRAYTESTMESSAGE"))

(in-package "ROS")
;;//! \htmlinclude Uint8ArrayTestMessage.msg.html


(defclass rospy_message_converter::Uint8ArrayTestMessage
  :super ros::object
  :slots (_data ))

(defmethod rospy_message_converter::Uint8ArrayTestMessage
  (:init
   (&key
    ((:data __data) (make-array 0 :initial-element 0 :element-type :char))
    )
   (send-super :init)
   (setq _data __data)
   self)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:serialization-length
   ()
   (+
    ;; uint8[] _data
    (* 1    (length _data)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8[] _data
     (write-long (length _data) s)
     (princ _data s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8[] _data
   (let ((n (sys::peek buf ptr- :integer))) (incf ptr- 4)
     (setq _data (make-array n :element-type :char))
     (replace _data buf :start2 ptr-) (incf ptr- n))
   ;;
   self)
  )

(setf (get rospy_message_converter::Uint8ArrayTestMessage :md5sum-) "f43a8e1b362b75baa741461b46adc7e0")
(setf (get rospy_message_converter::Uint8ArrayTestMessage :datatype-) "rospy_message_converter/Uint8ArrayTestMessage")
(setf (get rospy_message_converter::Uint8ArrayTestMessage :definition-)
      "# Size-agnostic uint8 array for testing purposes
uint8[] data

")



(provide :rospy_message_converter/Uint8ArrayTestMessage "f43a8e1b362b75baa741461b46adc7e0")


