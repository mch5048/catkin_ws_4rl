
(cl:in-package :asdf)

(defsystem "gazebo_ext_msgs-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "GetCollisionNames" :depends-on ("_package_GetCollisionNames"))
    (:file "_package_GetCollisionNames" :depends-on ("_package"))
    (:file "GetLinkVisualProperties" :depends-on ("_package_GetLinkVisualProperties"))
    (:file "_package_GetLinkVisualProperties" :depends-on ("_package"))
    (:file "GetSkyProperties" :depends-on ("_package_GetSkyProperties"))
    (:file "_package_GetSkyProperties" :depends-on ("_package"))
    (:file "GetSurfaceParams" :depends-on ("_package_GetSurfaceParams"))
    (:file "_package_GetSurfaceParams" :depends-on ("_package"))
    (:file "GetVisualNames" :depends-on ("_package_GetVisualNames"))
    (:file "_package_GetVisualNames" :depends-on ("_package"))
    (:file "SetLinkVisualProperties" :depends-on ("_package_SetLinkVisualProperties"))
    (:file "_package_SetLinkVisualProperties" :depends-on ("_package"))
    (:file "SetSkyProperties" :depends-on ("_package_SetSkyProperties"))
    (:file "_package_SetSkyProperties" :depends-on ("_package"))
    (:file "SetSurfaceParams" :depends-on ("_package_SetSurfaceParams"))
    (:file "_package_SetSurfaceParams" :depends-on ("_package"))
  ))