
(cl:in-package :asdf)

(defsystem "sawyer_sim_examples-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "GoalObs" :depends-on ("_package_GoalObs"))
    (:file "_package_GoalObs" :depends-on ("_package"))
    (:file "PosCmd" :depends-on ("_package_PosCmd"))
    (:file "_package_PosCmd" :depends-on ("_package"))
  ))