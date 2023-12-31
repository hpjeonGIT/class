## Title: Lisp Tutorial
- Instructor: Derek Banas
- Site: https://youtu.be/ymSq4wHrqyU?si=VeXpJTJvsG_uUrNI

- gnu lisp
  - sudo apt install gcl
  - gcl
  - To exit, `(bye)`
- SBCL
  - sudo apt install sbcl
  - sbcl
    - To exit, `(quit)`
- ;;;; Comment in the beginning
- ; comment in the middle of the code
- (format t "Hello world ~%")Top level.
```lisp
Hello World 
NIL

>(format t "(+ 5 4 ) = ~d ~%"(+ 5 4))
(+ 5 4 ) = 9 
NIL

>(format t "(/ 5 4 ) = ~d ~%"(/ 5 4))
(/ 5 4 ) = 5/4 
NIL

>(format t "(rem 5 4 ) = ~d ~%"(rem 5 4))
(rem 5 4 ) = 1 
NIL

>(format t "(mod 5 4 ) = ~d ~%"(mod 5 4))
(mod 5 4 ) = 1 
NIL
>>(defparameter *name* 'Derek)

*NAME*
>>(format t "(eq *name 'Derek) = ~d ~%" (eq *name* 'Derek))
(eq *name 'Derek) = T 
NIL
>>(format t "(eq *name 'Derek) = ~d ~%" (eq *name* 'Derek2))
(eq *name 'Derek) = NIL 
NIL
>>(equal 5 5)

T
>>(equal 5 5.0)

NIL
>(equalp 5 5.0)  ; can compare INT vs FLOAT

T
>(equal (list 1 2 3) (list 1 2 3))

T
>(defvar *age* 18) ; once defined, it cannot be changed

*AGE*

>(if (>= *age* 18)
  (format t "Can vote~%") ; executed when True
  (format t "Not yet~%")) ; executed when False/Nil
Can vote
NIL


```
- NIL == False
