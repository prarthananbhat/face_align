import cv2
import numpy as np
import dlib
import faceblendCommon as fbc
import matplotlib.pyplot as plt

faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Read Image
IMG_PATH = "/Users/prarthanabhat/Projects/2021/FaceAlign/face_align/"
imageFilename = IMG_PATH + "images/image001.jpeg"
img = cv2.imread(imageFilename)
plt.imshow(img[:, :, ::-1])
plt.show()

points = fbc.getLandmarks(faceDetector,landmarkDetector,img)
points = np.array(points,dtype=np.int32)
img = np.float32(img)/255.0

h=600
w =600
print(type(img))
print(type(points))
imnorm, points = fbc.normalizeImagesAndLandmarks((h,w),img,points)
imnorm = np.uint8(imnorm*255)

plt.imshow(imnorm[:,:,::-1])
plt.title("Alligned Image")
plt.show()