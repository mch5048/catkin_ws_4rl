;; Auto-generated. Do not edit!


(when (boundp 'berkeley_sawyer::save_kinectdata)
  (if (not (find-package "BERKELEY_SAWYER"))
    (make-package "BERKELEY_SAWYER"))
  (shadow 'save_kinectdata (find-package "BERKELEY_SAWYER")))
(unless (find-package "BERKELEY_SAWYER::SAVE_KINECTDATA")
  (make-package "BERKELEY_SAWYER::SAVE_KINECTDATA"))
(unless (find-package "BERKELEY_SAWYER::SAVE_KINECTDATAREQUEST")
  (make-package "BERKELEY_SAWYER::SAVE_KINECTDATAREQUEST"))
(unless (find-package "BERKELEY_SAWYER::SAVE_KINECTDATARESPONSE")
  (make-package "BERKELEY_SAWYER::SAVE_KINECTDATARESPONSE"))

(in-package "ROS")





(defclass berkeley_sawyer::save_kinectdataRequest
  :super ros::object
  :slots (_itr ))

(defmethod berkeley_sawyer::save_kinectdataRequest
  (:init
   (&key
    ((:itr __itr) 0)
    )
   (send-super :init)
   (setq _itr (round __itr))
   self)
  (:itr
   (&optional __itr)
   (if __itr (setq _itr __itr)) _itr)
  (:serialization-length
   ()
   (+
    ;; int64 _itr
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
   ;;
   self)
  )

(defclass berkeley_sawyer::save_kinectdataResponse
  :super ros::object
  :slots ())

(defmethod berkeley_sawyer::save_kinectdataResponse
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

(defclass berkeley_sawyer::save_kinectdata
  :super ros::object
  :slots ())

(setf (get berkeley_sawyer::save_kinectdata :md5sum-) "36618b1eaec8d2483d42a27cb2744012")
(setf (get berkeley_sawyer::save_kinectdata :datatype-) "berkeley_sawyer/save_kinectdata")
(setf (get berkeley_sawyer::save_kinectdata :request) berkeley_sawyer::save_kinectdataRequest)
(setf (get berkeley_sawyer::save_kinectdata :response) berkeley_sawyer::save_kinectdataResponse)

(defmethod berkeley_sawyer::save_kinectdataRequest
  (:response () (instance berkeley_sawyer::save_kinectdataResponse :init)))

(setf (get berkeley_sawyer::save_kinectdataRequest :md5sum-) "36618b1eaec8d2483d42a27cb2744012")
(setf (get berkeley_sawyer::save_kinectdataRequest :datatype-) "berkeley_sawyer/save_kinectdataRequest")
(setf (get berkeley_sawyer::save_kinectdataRequest :definition-)
      "int64 itr
---
")

(setf (get berkeley_sawyer::save_kinectdataResponse :md5sum-) "36618b1eaec8d2483d42a27cb2744012")
(setf (get berkeley_sawyer::save_kinectdataResponse :datatype-) "berkeley_sawyer/save_kinectdataResponse")
(setf (get berkeley_sawyer::save_kinectdataResponse :definition-)
      "int64 itr
---
")



(provide :berkeley_sawyer/save_kinectdata "36618b1eaec8d2483d42a27cb2744012")


