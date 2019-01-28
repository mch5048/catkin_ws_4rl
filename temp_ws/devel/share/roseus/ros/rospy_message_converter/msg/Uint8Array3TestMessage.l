;; Auto-generated. Do not edit!


(when (boundp 'rospy_message_converter::Uint8Array3TestMessage)
  (if (not (find-package "ROSPY_MESSAGE_CONVERTER"))
    (make-package "ROSPY_MESSAGE_CONVERTER"))
  (shadow 'Uint8Array3TestMessage (find-package "ROSPY_MESSAGE_CONVERTER")))
(unless (find-package "ROSPY_MESSAGE_CONVERTER::UINT8ARRAY3TESTMESSAGE")
  (make-package "ROSPY_MESSAGE_CONVERTER::UINT8ARRAY3TESTMESSAGE"))

(in-package "ROS")
;;//! \htmlinclude Uint8Array3TestMessage.msg.html


(defclass rospy_message_converter::Uint8Array3TestMessage
  :super ros::object
  :slots (_data ))

(defmethod rospy_message_converter::Uint8Array3TestMessage
  (:init
   (&key
    ((:data __data) (make-array 3 :initial-element 0 :element-type :char))
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
    ;; uint8[3] _data
    (* 1    3)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8[3] _data
     (princ _data s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8[3] _data
   (setq _data (make-array 3 :element-type :char))
   (replace _data buf :start2 ptr-) (incf ptr- 3)
   ;;
   self)
  )

(setf (get rospy_message_converter::Uint8Array3TestMessage :md5sum-) "0aff379edf54a041a83af067122e7bc6")
(setf (get rospy_message_converter::Uint8Array3TestMessage :datatype-) "rospy_message_converter/Uint8Array3TestMessage")
(setf (get rospy_message_converter::Uint8Array3TestMessage :definition-)
      "# Fixed size uint8 array for testing purposes
uint8[3] data

")



(provide :rospy_message_converter/Uint8Array3TestMessage "0aff379edf54a041a83af067122e7bc6")


