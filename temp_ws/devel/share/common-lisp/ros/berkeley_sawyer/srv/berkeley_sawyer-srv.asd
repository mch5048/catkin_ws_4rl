
(cl:in-package :asdf)

(defsystem "berkeley_sawyer-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :sensor_msgs-msg
)
  :components ((:file "_package")
    (:file "delete_traj" :depends-on ("_package_delete_traj"))
    (:file "_package_delete_traj" :depends-on ("_package"))
    (:file "get_action" :depends-on ("_package_get_action"))
    (:file "_package_get_action" :depends-on ("_package"))
    (:file "get_kinectdata" :depends-on ("_package_get_kinectdata"))
    (:file "_package_get_kinectdata" :depends-on ("_package"))
    (:file "init_traj" :depends-on ("_package_init_traj"))
    (:file "_package_init_traj" :depends-on ("_package"))
    (:file "init_traj_visualmpc" :depends-on ("_package_init_traj_visualmpc"))
    (:file "_package_init_traj_visualmpc" :depends-on ("_package"))
    (:file "save_kinectdata" :depends-on ("_package_save_kinectdata"))
    (:file "_package_save_kinectdata" :depends-on ("_package"))
  ))