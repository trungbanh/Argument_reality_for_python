### hi my name is Banh Phuoc Trung this is my small project 
# Argument_reality_for_python

to complete it i have consult some tutorial [Augmented reality with Python and OpenCV](https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/) and [programming computer vision with python](http://programmingcomputervision.com/downloads/ProgrammingComputerVision_CCdraft.pdf) thanks Jan Erik Solem because of this book. 
in this project i have read many times of this to understand idea and math to make it. 

So get start

## first setup camera calibration
opencv have a small code to do it [opencv-doc](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html) 
the result of 
```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
``` 

save `mtx` for late use  

## searching image mode
next i use OBR of opencv to get feature detect 

```python
orb = cv2.ORB_create()
model = cv2.imread('model.jpg', 0)
model = cv2.resize(model, (int(width/1.2), int(height/1.2)))
kp1, des1 = sift.detectAndCompute(model, None)
```
Do same with image in video input i have 2 key point and 2 descriptors 

now we can match it with 
```python
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
```

matches is array of index of keypoint matched,
we use it to find best key point to find homography
```python
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
```

to here we can see how to findout mode image in video
## add agrument to reality
cause im not study well in computer graphic that i use already function of OpenGl mentioned in `programming computer vision with python` 

```python 
def draw_3D():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0, 0, 0.1, 0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0, 0, 0])
    glMaterialfv(GL_FRONT, GL_SHININESS, 0.25 * 128)
    glutSolidTeapot(0.1)
```
## merge image 3D with input video

```ketqua = cv2.addWeighted(frame_of_video, 1, image_of_model, 1, 1)```

all done 

## to use my code 
> pip3 install -r requirement.txt

> python3 3dpygl.py 
