
(cl:in-package :asdf)

(defsystem "ddpg-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "GoalObs" :depends-on ("_package_GoalObs"))
    (:file "_package_GoalObs" :depends-on ("_package"))
  ))