from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import math
from scipy import ndimage
from pyzbar.pyzbar import decode


def barcode_qr():
    img = cv2.imread("../tests/input/0008.jpg")
    barcodes = decode(img)
    for barcode in barcodes:
        print(barcode)
        pt1_point = [barcode.rect.left, barcode.rect.top]
        pt2_point = [
            pt1_point[0] + barcode.rect.width,
            pt1_point[1] + barcode.rect.height,
        ]
        cv2.rectangle(img, pt1_point, pt2_point, (255, 0, 0), 5)
        # nr_of_points = len(barcode.polygon)
        # for i  in range(nr_of_points):
        #     next_point_index = (i + 1) % nr_of_points
        #     print(i, next_point_index)
        #     cv2.line(img, barcode.polygon[i], barcode.polygon[next_point_index], (255, 0, 0), 1*(i+1))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
        # cv2.imshow("Image", img)
        # cv2.waitKey(10000)
        # cv2.destroyAllWindows()


def barcode_qr_cv():
    debug = True
    print(cv2.__version__)

    img = cv2.imread("../tests/input/250550361.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    qr_code_detector = cv2.QRCodeDetector()
    # img = cv2.flip(img, 1)
    decoded_text, points, _ = qr_code_detector.detectAndDecode(img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    if points is not None:
        nr_of_points = len(points)

        for i in range(nr_of_points):
            next_point_index = (i + 1) % nr_of_points
            print(points[i])
            # cv2.line(img, tuple(points[i][0]), tuple(points[next_point_index][0]), (255, 0, 0), 5)

        print(decoded_text)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # cv2.imshow("Image", img)
        # cv2.waitKey(10000)
        # cv2.destroyAllWindows()
        print("QR code ", points, decoded_text)
    else:
        print("QR code not detected", decoded_text)


def barcode_linear_cv():
    debug = True
    print(cv2.__version__)

    img = cv2.imread("../tests/input/0001.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bd = cv2.barcode.BarcodeDetector()
    # bd = cv2.barcode.BarcodeDetector('sr.prototxt', 'sr.caffemodel')
    # img = cv2.flip(img, 1)
    retval, decoded_info, decoded_type, _ = bd.detectAndDecodeWithType(img)
    print(retval, decoded_info, decoded_type)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    return

    if points is not None:
        nr_of_points = len(points)

        for i in range(nr_of_points):
            next_point_index = (i + 1) % nr_of_points
            print(points[i])
            # cv2.line(img, tuple(points[i][0]), tuple(points[next_point_index][0]), (255, 0, 0), 5)

        print(decoded_text)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # cv2.imshow("Image", img)
        # cv2.waitKey(10000)
        # cv2.destroyAllWindows()
        print("QR code ", points, decoded_text)
    else:
        print("QR code not detected", decoded_text)


def im_scan_barcode(
    file_path: Path, output: Path, parameters=None, debug: bool = False
):
    # print(f"STILL FAKE. Just print :) {__package__}, im_scan {file_path}")
    if parameters is None:
        parameters = {}

    size = file_path.stat().st_size
    date_m = datetime.fromtimestamp(file_path.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M"
    )
    modified = str(date_m)
    print(f"File: '{file_path.name}' {size=} bytes, {modified=}")
    return barcode_scan(file_path, output, parameters=parameters, debug=debug)


def barcode_scan(file_path: Path, output: Path, parameters=None, debug: bool = False):
    print(cv2.__version__)
    success, warn = False, False
    if parameters is None:
        parameters = {}
    input_file: str = str(file_path)
    output_file: str = str(output.joinpath(file_path.name))
    # bd = cv2.barcode.BarcodeDetector()
    img = cv2.imread(input_file)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using OpenCV 2.4
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # cv2.imshow("Image gradient", gradient)
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
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    # cv2.imshow("Image cloed", closed)
    # plt.imshow(closed)
    # plt.show()

    # cnts0, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.intp(box)
    # draw a bounding box arounded the detected barcode and display the
    # image
    if debug:
        img_d = img.copy()

        # cv2.drawContours(img_d, cnts0, -1, (255, 0, 0), 3)  # in blue
        cv2.drawContours(img_d, [box], -1, (0, 255, 0), 1)

        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite("result.png", img_d)
        plt.figure(1)
        plt_img_rows = 1
        plt_img_cols = 4
        plt_img = 1
        plt.subplot(plt_img_rows, plt_img_cols, plt_img)
        plt.imshow(cv2.cvtColor(closed, cv2.COLOR_BGR2RGB))
        plt_img += 1
        plt.subplot(plt_img_rows, plt_img_cols, plt_img)
        plt.imshow(cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB))

    (x, y), (h, w), angle = rect
    x = math.ceil(x)
    y = math.ceil(y)
    h = math.ceil(h)
    w = math.ceil(w)

    height, width = img.shape[:2]

    print(f"box: {x=} {y=} {w=} {h=}  img: {width=} x {height=}  ")

    if width < 300 or height < 100:
        print("box: not found ")
        return success, warn

    aspect = w / h
    aspect_ideal = 3.9386
    aspect_corrected = aspect / aspect_ideal

    corr_bar_size_width = w
    corr_bar_size_height = h * aspect_corrected * 1.02

    corr_img_size_width = width
    corr_img_size_height = height * aspect_corrected

    print(f"corr bar: {corr_bar_size_width=} x {corr_bar_size_height=}  ")
    print(f"corr img: {corr_img_size_width=} x {corr_img_size_height=}  ")

    M = cv2.moments(box)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    print(f"box: {cX=} {cY=} ")
    angle_rot = -(90 - angle)
    center = (cX, cY)
    rotMat = cv2.getRotationMatrix2D(
        center, angle_rot, 1.0
    )  # Get the rotation matrix, its of shape 2x3
    img_rotated = cv2.warpAffine(img, rotMat, img.shape[1::-1])  # Rotate the image

    dim = (int(corr_img_size_width), int(corr_img_size_height))
    img_rotated = cv2.resize(img_rotated, dim)
    cY = cY * aspect_corrected

    border_size = 1000
    mean = 0

    img_rotated_crop = cv2.copyMakeBorder(
        img_rotated,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean],
    )

    cY = cY + border_size
    cX = cX + border_size

    top = 6.7368 * corr_bar_size_height
    bottom = 7.78 * corr_bar_size_height
    left = 1.3363 * corr_bar_size_width
    rigth = 1.5145 * corr_bar_size_width

    # br = 5
    # top = br + corr_bar_size_height / 2
    # bottom = br + corr_bar_size_height / 2
    # left = br + corr_bar_size_width / 2
    # rigth = br + corr_bar_size_width / 2

    print(f"CROP AREA: {top=} {bottom=} {left=} {rigth=}")

    img_rotated_crop = img_rotated_crop[
        int(cY - top) : int(cY + bottom), int(cX - left) : int(cX + rigth)
    ]

    # row, col = img_rotated_crop.shape[:2]
    # bottom = img_rotated_crop[row - 2:row, 0:col]

    # rotated = ndimage.rotate(img, -(90-angle), cval=255)

    if debug:
        plt_img += 1
        plt.subplot(plt_img_rows, plt_img_cols, plt_img)
        plt.imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
        plt_img += 1
        plt.subplot(plt_img_rows, plt_img_cols, plt_img)
        plt.imshow(cv2.cvtColor(img_rotated_crop, cv2.COLOR_BGR2RGB))
        # plt.suptitle('IMG')
        # plt.subplots_adjust(wspace=0, hspace=0)

        plt.show()
        # plt.draw()
        # plt.waitforbuttonpress(40)

    cv2.imwrite(output_file, img_rotated_crop)
    # leave memory
    img_rotated_crop = np.zeros(0)
    img_rotated = np.zeros(0)
    img = np.zeros(0)
    gray = np.zeros(0)
    success = True
    return success, warn


def barcode_01(file_path: Path, output: Path, parameters=None, debug: bool = False):
    print(cv2.__version__)
    bd = cv2.barcode.BarcodeDetector()
    # bd = cv2.barcode.BarcodeDetector('sr.prototxt', 'sr.caffemodel')
    img = cv2.imread("../tests/input/0001.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using OpenCV 2.4
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # cv2.imshow("Image gradient", gradient)
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
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    # cv2.imshow("Image cloed", closed)
    # plt.imshow(closed)
    # plt.show()

    # cnts0, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.intp(box)
    # draw a bounding box arounded the detected barcode and display the
    # image
    img_d = img.copy()

    # cv2.drawContours(img_d, cnts0, -1, (255, 0, 0), 3)  # in blue
    cv2.drawContours(img_d, [box], -1, (0, 255, 0), 1)

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if debug:
        cv2.imwrite("result.png", img_d)
        plt.figure(1)
        plt_img_rows = 1
        plt_img_cols = 4
        plt_img = 1
        plt.subplot(plt_img_rows, plt_img_cols, plt_img)
        plt.imshow(cv2.cvtColor(closed, cv2.COLOR_BGR2RGB))
        plt_img += 1
        plt.subplot(plt_img_rows, plt_img_cols, plt_img)
        plt.imshow(cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB))

    (x, y), (h, w), angle = rect
    x = math.ceil(x)
    y = math.ceil(y)
    h = math.ceil(h)
    w = math.ceil(w)

    height, width = img.shape[:2]

    print(f"box: {x=} {y=} {w=} {h=}  img: {width=} x {height=}  ")
    aspect = w / h
    aspect_ideal = 3.9386
    aspect_corrected = aspect / aspect_ideal

    corr_bar_size_width = w
    corr_bar_size_height = h * aspect_corrected * 1.02

    corr_img_size_width = width
    corr_img_size_height = height * aspect_corrected

    print(f"corr bar: {corr_bar_size_width=} x {corr_bar_size_height=}  ")
    print(f"corr img: {corr_img_size_width=} x {corr_img_size_height=}  ")

    M = cv2.moments(box)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    print(f"box: {cX=} {cY=} ")
    angle_rot = -(90 - angle)
    center = (cX, cY)
    rotMat = cv2.getRotationMatrix2D(
        center, angle_rot, 1.0
    )  # Get the rotation matrix, its of shape 2x3
    img_rotated = cv2.warpAffine(img, rotMat, img.shape[1::-1])  # Rotate the image

    dim = (int(corr_img_size_width), int(corr_img_size_height))
    img_rotated = cv2.resize(img_rotated, dim)
    cY = cY * aspect_corrected

    border_size = 1000
    mean = 0

    img_rotated_crop = cv2.copyMakeBorder(
        img_rotated,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean],
    )

    cY = cY + border_size
    cX = cX + border_size

    top = 6.7368 * corr_bar_size_height
    bottom = 7.78 * corr_bar_size_height
    left = 1.3363 * corr_bar_size_width
    rigth = 1.5145 * corr_bar_size_width

    # br = 5
    # top = br + corr_bar_size_height / 2
    # bottom = br + corr_bar_size_height / 2
    # left = br + corr_bar_size_width / 2
    # rigth = br + corr_bar_size_width / 2

    print(f"CROP AREA: {top=} {bottom=} {left=} {rigth=}")

    img_rotated_crop = img_rotated_crop[
        int(cY - top) : int(cY + bottom), int(cX - left) : int(cX + rigth)
    ]

    # row, col = img_rotated_crop.shape[:2]
    # bottom = img_rotated_crop[row - 2:row, 0:col]

    # rotated = ndimage.rotate(img, -(90-angle), cval=255)

    if debug:
        cv2.imwrite("../tests/input/rot-crop-result.png", img_rotated_crop)
        plt_img += 1
        plt.subplot(plt_img_rows, plt_img_cols, plt_img)
        plt.imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
        plt_img += 1
        plt.subplot(plt_img_rows, plt_img_cols, plt_img)
        plt.imshow(cv2.cvtColor(img_rotated_crop, cv2.COLOR_BGR2RGB))
        # plt.suptitle('IMG')
        # plt.subplots_adjust(wspace=0, hspace=0)

        plt.show()
        # plt.draw()
        # plt.waitforbuttonpress(40)


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

# decoded_info, decoded_type, points = bd.detectAndDecode(img)
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


if __name__ == "__main__":
    barcode_01()
    # barcode_qr()
    # barcode_qr_cv()
    # barcode_linear_cv()
