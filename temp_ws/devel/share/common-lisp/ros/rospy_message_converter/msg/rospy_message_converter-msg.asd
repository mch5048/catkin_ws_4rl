
(cl:in-package :asdf)

(defsystem "rospy_message_converter-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "TestArray" :depends-on ("_package_TestArray"))
    (:file "_package_TestArray" :depends-on ("_package"))
    (:file "Uint8Array3TestMessage" :depends-on ("_package_Uint8Array3TestMessage"))
    (:file "_package_Uint8Array3TestMessage" :depends-on ("_package"))
    (:file "Uint8ArrayTestMessage" :depends-on ("_package_Uint8ArrayTestMessage"))
    (:file "_package_Uint8ArrayTestMessage" :depends-on ("_package"))
  ))