import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import math
from scipy import ndimage

debug = True
print(cv2.__version__)
bd = cv2.barcode.BarcodeDetector()
# bd = cv2.barcode.BarcodeDetector('sr.prototxt', 'sr.caffemodel')
img = cv2.imread('../tests/input/0001.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction using OpenCV 2.4
ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

#cv2.imshow("Image gradient", gradient)
# plt.imshow(gradient)
# plt.show()

# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

# plt.imshow(thresh)
# plt.show()

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# plt.imshow(closed)
# plt.show()

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)

#cv2.imshow("Image cloed", closed)
# plt.imshow(closed)
# plt.show()

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.intp(box)
# draw a bounding box arounded the detected barcode and display the
# image
img_d = img.copy()
cv2.drawContours(img_d, [box], -1, (0, 255, 0), 3)
cv2.imwrite("result.png", img_d)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
if debug:
    plt.figure(1)
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB))


(x,y),(h,w),angle = rect
x = math.ceil(x)
y = math.ceil(y)
h = math.ceil(h)
w = math.ceil(w)

height, width = img.shape[:2]

print(f"box: {x=} {y=} {w=} {h=}  img: {width=} x {height=}  ")

M = cv2.moments(box)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

print(f"box: {cX=} {cY=} ")
angle_rot = -(90-angle)
center = (cX,cY)
rotMat = cv2.getRotationMatrix2D(center, angle_rot, 1.0)  # Get the rotation matrix, its of shape 2x3
img_rotated = cv2.warpAffine(img, rotMat, img.shape[1::-1])  # Rotate the image

top=470
bottom=820
left=490
rigth=580

img_rotated_crop = img_rotated[int(cY - top):int(cY + bottom), int(cX - left):int(cX + rigth)]

border_size = 40

row, col = img_rotated_crop.shape[:2]
bottom = img_rotated_crop[row-2:row, 0:col]
mean = 255

img_rotated_crop = cv2.copyMakeBorder(
    img_rotated_crop,
    top=border_size,
    bottom=border_size,
    left=border_size,
    right=border_size,
    borderType=cv2.BORDER_CONSTANT,
    value=[mean, mean, mean]
)



#rotated = ndimage.rotate(img, -(90-angle), cval=255)

if debug:
    cv2.imwrite("../tests/input/rot-crop-result.png", img_rotated_crop)
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(img_rotated_crop, cv2.COLOR_BGR2RGB))
    plt.suptitle('IMG')
    plt.show()


# if False:
#     (rv, detections) = bd.detect(img)
#     print(detections)
#     for barcode in detections:
#         cv2.polylines(img, [np.int32(barcode)], isClosed=True, color=(0, 0, 255), thickness=2)
#
#
#     cv2.imshow("IMG",img)
#     cv2.waitKey(5000)
#     cv2.destroyAllWindows()

#decoded_info, decoded_type, points = bd.detectAndDecode(img)
#
#
#
# print(decoded_info)
# # ('1923055034006', '9784873117980')
#
# print(decoded_type)
# # (2, 2)
#
#
#
# print(type(points))
# # <class 'numpy.ndarray'>
