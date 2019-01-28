
(cl:in-package :asdf)

(defsystem "visual_mpc_rospkg-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :sensor_msgs-msg
)
  :components ((:file "_package")
    (:file "get_action" :depends-on ("_package_get_action"))
    (:file "_package_get_action" :depends-on ("_package"))
    (:file "init_traj_visualmpc" :depends-on ("_package_init_traj_visualmpc"))
    (:file "_package_init_traj_visualmpc" :depends-on ("_package"))
  ))