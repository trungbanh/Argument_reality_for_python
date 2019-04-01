
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import pygame
import pygame.image
from pygame.locals import *
import numpy as np
import pickle
import cv2

import camera
import homography


width, height = 1280, 720
sift = cv2.ORB_create()
model = cv2.imread('model.jpg', 0)
model = cv2.resize(model, (int(width/1), int(height/1)))
kp1, des1 = sift.detectAndCompute(model, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# K = np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]])
K = np.array([[1186, 0, 656], [0, 1168, 380], [0, 0, 1]])


def set_projection_from_camera(K):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    fovy = 2 * np.arctan(0.5 * height / fy) * 180 / np.pi
    aspect = (width * fy) / (height * fx)

    near, far = 0.1, 100
    gluPerspective(fovy, aspect, near, far)
    glViewport(0, 0, width, height)


def set_modelview_from_camera(Rt):
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Rotate 90 deg around x, so that z is up.
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    try:
        global m
        R = Rt[:, :3]
        U, S, V = np.linalg.svd(R)
        R = np.dot(U, V)
        R[0, :] = -R[0, :]  # Change sign of x axis.

        t = Rt[:, 3]

        M = np.eye(4)
        M[:3, :3] = np.dot(R, Rx)
        M[:3, 3] = t

        m = M.T.flatten()
        # print (type(m))
    except TypeError as e:
        pass
    glLoadMatrixf(m)


def setup():
    '''
    khoi tao khung hinh cua opengl thong qua pygame
    '''
    pygame.init()
    pygame.display.set_mode((width, height), OPENGL | DOUBLEBUF)
    pygame.display.set_caption('OpenGL window')
    glutInit(sys.argv)


def draw_3D():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)
    glClear(GL_COLOR_BUFFER_BIT)
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0, 0, 0, 0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0, 0, 0])
    glMaterialfv(GL_FRONT, GL_SHININESS, 0.25 * 128)

    glutSolidTeapot(0.1)


def getRt(image):
    '''
    sift
    kp1, des1
    bf
    '''
    kp2, des2 = sift.detectAndCompute(image, None)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    good = []
    for m in matches:
        if m.distance > 0.7:
            good.append(m)

    if len(good) > 70:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        cam1 = camera.Camera(
            np.hstack((K, np.dot(K, np.array([[0], [0], [-1]])))))

        cam2 = camera.Camera(np.dot(H, cam1.P))

        return np.dot(np.linalg.inv(K), cam2.P)


if __name__ == "__main__":

    setup()
    video = cv2.VideoCapture('/home/trungbanh/tam/video.mp4')
    ketqua = np.zeros((width, height, 4))

    # out = cv2.VideoWriter('outpy.mp4', cv2.VideoWriter_fourcc(
    # 'M', 'J', 'P', 'G'), 30, (width, height))

    while video.isOpened():
        pygame.display.iconify()

        _, img = video.read()
        img = cv2.cvtColor(img, cv2.cv2.COLOR_RGB2RGBA)
        img = cv2.resize(img, (width, height))

        Rt = getRt(img)

        # thiet lap ma tran camera va homography
        set_projection_from_camera(K)
        set_modelview_from_camera(Rt)
        draw_3D()

        # trich anh tu GL
        image_buffer = glReadPixels(
            0, 0, width, height, OpenGL.GL.GL_RGBA, OpenGL.GL.GL_UNSIGNED_BYTE)
        image = np.frombuffer(
            image_buffer, dtype=np.uint8).reshape(height, width, 4)

        # chuyen anh ve numpy array de tinh toan nhanh hon
        image = cv2.cvtColor(image, cv2.cv2.COLOR_BGRA2RGBA)
        image = np.array(image)
        image = np.flip(image, 0)

        tam = np.where(image[:, :, 2] != 0, 0, 1)

        img[:, :, 0] = img[:, :, 0] * tam
        img[:, :, 1] = img[:, :, 1] * tam
        img[:, :, 2] = img[:, :, 2] * tam

        ketqua = cv2.addWeighted(image, 1, img, 1, 10)
        cv2.imshow('image', ketqua)
        # out.write(ketqua)

        # pygame flip chuyen canh
        pygame.display.flip()
        event = pygame.event.poll()
        if event.type == pygame.QUIT or cv2.waitKey(1) & 0xFF == ord('q'):
            pygame.quit()
            break

    video.release()
    cv2.cv2.destroyAllWindows()
