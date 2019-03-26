### hi my name is Banh Phuoc Trung this is my small project 
# Argument_reality_for_python

to complete it i have consult some tutorial [Augmented reality with Python and OpenCV](https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/) and [programming computer vision with python](http://programmingcomputervision.com/downloads/ProgrammingComputerVision_CCdraft.pdf) thanks Jan Erik Solem because of this book. 
in this project i have read many times of this to understand idea and math to make it. so get start

##first setup camera calibration
opencv have a small code to do it [opencv-doc](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html) 
the result of `
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)`
mtx is that one we need it look like $ \left ( \begin{array}{ccc}
f_{x} & 0 & s_{1} \\
0 & f_{y} & s_{2} \\
0 & 0 & 1 \\
\end{array} \right ) $

next i use OBR of opencv to get feature detect 
