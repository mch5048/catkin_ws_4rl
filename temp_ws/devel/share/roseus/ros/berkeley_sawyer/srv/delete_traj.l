;; Auto-generated. Do not edit!


(when (boundp 'berkeley_sawyer::delete_traj)
  (if (not (find-package "BERKELEY_SAWYER"))
    (make-package "BERKELEY_SAWYER"))
  (shadow 'delete_traj (find-package "BERKELEY_SAWYER")))
(unless (find-package "BERKELEY_SAWYER::DELETE_TRAJ")
  (make-package "BERKELEY_SAWYER::DELETE_TRAJ"))
(unless (find-package "BERKELEY_SAWYER::DELETE_TRAJREQUEST")
  (make-package "BERKELEY_SAWYER::DELETE_TRAJREQUEST"))
(unless (find-package "BERKELEY_SAWYER::DELETE_TRAJRESPONSE")
  (make-package "BERKELEY_SAWYER::DELETE_TRAJRESPONSE"))

(in-package "ROS")





(defclass berkeley_sawyer::delete_trajRequest
  :super ros::object
  :slots (_itr _igrp ))

(defmethod berkeley_sawyer::delete_trajRequest
  (:init
   (&key
    ((:itr __itr) 0)
    ((:igrp __igrp) 0)
    )
   (send-super :init)
   (setq _itr (round __itr))
   (setq _igrp (round __igrp))
   self)
  (:itr
   (&optional __itr)
   (if __itr (setq _itr __itr)) _itr)
  (:igrp
   (&optional __igrp)
   (if __igrp (setq _igrp __igrp)) _igrp)
  (:serialization-length
   ()
   (+
    ;; int64 _itr
    8
    ;; int64 _igrp
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int64 _itr
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _itr (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _itr) (= (length (_itr . bv)) 2)) ;; bignum
              (write-long (ash (elt (_itr . bv) 0) 0) s)
              (write-long (ash (elt (_itr . bv) 1) -1) s))
             ((and (class _itr) (= (length (_itr . bv)) 1)) ;; big1
              (write-long (elt (_itr . bv) 0) s)
              (write-long (if (>= _itr 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _itr s)(write-long (if (>= _itr 0) 0 #xffffffff) s)))
     ;; int64 _igrp
#+(or :alpha :irix6 :x86_64)
       (progn (sys::poke _igrp (send s :buffer) (send s :count) :long) (incf (stream-count s) 8))
#-(or :alpha :irix6 :x86_64)
       (cond ((and (class _igrp) (= (length (_igrp . bv)) 2)) ;; bignum
              (write-long (ash (elt (_igrp . bv) 0) 0) s)
              (write-long (ash (elt (_igrp . bv) 1) -1) s))
             ((and (class _igrp) (= (length (_igrp . bv)) 1)) ;; big1
              (write-long (elt (_igrp . bv) 0) s)
              (write-long (if (>= _igrp 0) 0 #xffffffff) s))
             (t                                         ;; integer
              (write-long _igrp s)(write-long (if (>= _igrp 0) 0 #xffffffff) s)))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int64 _itr
#+(or :alpha :irix6 :x86_64)
      (setf _itr (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _itr (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;; int64 _igrp
#+(or :alpha :irix6 :x86_64)
      (setf _igrp (prog1 (sys::peek buf ptr- :long) (incf ptr- 8)))
#-(or :alpha :irix6 :x86_64)
      (setf _igrp (let ((b0 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4)))
                  (b1 (prog1 (sys::peek buf ptr- :integer) (incf ptr- 4))))
              (cond ((= b1 -1) b0)
                     ((and (= b1  0)
                           (<= lisp::most-negative-fixnum b0 lisp::most-positive-fixnum))
                      b0)
                    ((= b1  0) (make-instance bignum :size 1 :bv (integer-vector b0)))
                    (t (make-instance bignum :size 2 :bv (integer-vector b0 (ash b1 1)))))))
   ;;
   self)
  )

(defclass berkeley_sawyer::delete_trajResponse
  :super ros::object
  :slots ())

(defmethod berkeley_sawyer::delete_trajResponse
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass berkeley_sawyer::delete_traj
  :super ros::object
  :slots ())

(setf (get berkeley_sawyer::delete_traj :md5sum-) "58b1b06ba616229bc38a06a0a0af5730")
(setf (get berkeley_sawyer::delete_traj :datatype-) "berkeley_sawyer/delete_traj")
(setf (get berkeley_sawyer::delete_traj :request) berkeley_sawyer::delete_trajRequest)
(setf (get berkeley_sawyer::delete_traj :response) berkeley_sawyer::delete_trajResponse)

(defmethod berkeley_sawyer::delete_trajRequest
  (:response () (instance berkeley_sawyer::delete_trajResponse :init)))

(setf (get berkeley_sawyer::delete_trajRequest :md5sum-) "58b1b06ba616229bc38a06a0a0af5730")
(setf (get berkeley_sawyer::delete_trajRequest :datatype-) "berkeley_sawyer/delete_trajRequest")
(setf (get berkeley_sawyer::delete_trajRequest :definition-)
      "int64 itr
int64 igrp
---

")

(setf (get berkeley_sawyer::delete_trajResponse :md5sum-) "58b1b06ba616229bc38a06a0a0af5730")
(setf (get berkeley_sawyer::delete_trajResponse :datatype-) "berkeley_sawyer/delete_trajResponse")
(setf (get berkeley_sawyer::delete_trajResponse :definition-)
      "int64 itr
int64 igrp
---

")



(provide :berkeley_sawyer/delete_traj "58b1b06ba616229bc38a06a0a0af5730")


