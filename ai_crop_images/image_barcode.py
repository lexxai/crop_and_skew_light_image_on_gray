import multiprocessing
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import math

# from scipy import ndimage
# from pyzbar.pyzbar import decode
# from PIL import Image, ImageDraw

import logging

# logger = logging.getLogger(__name__)
logger: logging


def init_logger(_name: str = None):
    global logger
    name = "" if _name is None else f".{_name}"
    logger = logging.getLogger(f"{__name__}{name}")


# def barcode_pyzbar(
#     file_path: Path, output: Path, parameters=None, debug: bool = False
# ) -> (bool, bool):
#     # img = cv2.imread(str(file_path))
#     img = Image.open(str(file_path)).convert("RGB")
#     barcodes = decode(img)
#     for barcode in barcodes:
#         if barcode.type == "CODE39" and barcode.quality >= 2:
#             result: str = barcode.data.decode()
#             poligon = barcode.polygon
#             for x, y in poligon:
#                 print(x, y)
#             print(barcode)
#             pt1_point = [barcode.rect.left, barcode.rect.top]
#             pt2_point = [
#                 pt1_point[0] + barcode.rect.width,
#                 pt1_point[1] + barcode.rect.height,
#             ]
#             # cv2.rectangle(img, pt1_point, pt2_point, (255, 0, 0), 5)
#             draw = ImageDraw.Draw(img)
#             rect = barcode.rect
#             draw.rectangle(
#                 (
#                     (rect.left, rect.top),
#                     (rect.left + rect.width, rect.top + rect.height),
#                 ),
#                 outline="#0080ff",
#             )
#
#             draw.polygon(barcode.polygon, outline="#e945ff", width=6)
#             # img.save("bounding_box_and_polygon.png")
#             img.show()
#             # cv2.polylines(img, poligon, False, (0, 255, 0), 5)
#             # draw.polygon(barcode.polygon, outline="#e945ff")
#             # nr_of_points = len(barcode.polygon)
#             # for i  in range(nr_of_points):
#             #     next_point_index = (i + 1) % nr_of_points
#             #     print(i, next_point_index)
#             #     cv2.line(img, barcode.polygon[i], barcode.polygon[next_point_index], (255, 0, 0), 1*(i+1))
#             # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             # plt.show()
#             # cv2.imshow("Image", img)
#             # cv2.waitKey(10000)
#             # cv2.destroyAllWindows()
#             return True, False
#     return False, False


# def barcode_qr_cv():
#     debug = True
#     print(cv2.__version__)
#
#     img = cv2.imread("../tests/input/250550361.jpg")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     qr_code_detector = cv2.QRCodeDetector()
#     # img = cv2.flip(img, 1)
#     decoded_text, points, _ = qr_code_detector.detectAndDecode(img)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.show()
#
#     if points is not None:
#         nr_of_points = len(points)
#
#         for i in range(nr_of_points):
#             next_point_index = (i + 1) % nr_of_points
#             print(points[i])
#             # cv2.line(img, tuple(points[i][0]), tuple(points[next_point_index][0]), (255, 0, 0), 5)
#
#         print(decoded_text)
#
#         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         # cv2.imshow("Image", img)
#         # cv2.waitKey(10000)
#         # cv2.destroyAllWindows()
#         print("QR code ", points, decoded_text)
#     else:
#         print("QR code not detected", decoded_text)


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(cnt, angle):
    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated


# def rotate_contour(cnt, angle):
#     if angle < -45:
#         angle = 90 + angle
#     # otherwise, just take the inverse of the angle to make
#     # it positive
#     else:
#         angle = -angle
#
#     M = cv2.moments(cnt)
#     cx = int(M["m10"] / M["m00"])
#     cy = int(M["m01"] / M["m00"])
#     center = (cx, cy)
#     rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
#     rotated_image = cv2.warpAffine(src=cnt, M=rotate_matrix)
#     return rotated_image


def scale_contour(cnt, scale_x, scale_y=None):
    if scale_y is None:
        scale_y = scale_x
    M = cv2.moments(cnt)
    cx: int = int(M["m10"] / M["m00"])
    cy: int = int(M["m01"] / M["m00"])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * (scale_x, scale_y)
    cnt_scaled = cnt_scaled + (cx, cy)
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def im_scan_barcode(
    file_path: Path,
    output: Path,
    parameters=None,
    debug: bool = False,
    barcode_method: int = 0,
    queue: multiprocessing.Queue = None,
    configurer=None,
) -> dict:
    if parameters is None:
        parameters = {}
    result = {"success": False, "warn": False, "im": file_path}
    success, warn = False, False
    if configurer is not None:
        configurer(queue)
    init_logger(file_path.name)
    size = file_path.stat().st_size
    date_m = datetime.fromtimestamp(file_path.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M"
    )
    modified = str(date_m)
    logger.debug(
        f"File: '{file_path.name}' {size=} bytes, {modified=} {barcode_method=}"
    )
    try:
        if barcode_method == 2:
            success, warn = barcode_linear_cv(
                file_path, output, parameters=parameters, debug=debug
            )
        elif barcode_method == 1:
            success, warn = barcode_scan(
                file_path, output, parameters=parameters, debug=debug
            )

        # success, warn = barcode_pyzbar(
        #     file_path, output, parameters=parameters, debug=debug
        # )
    except Exception as e:
        logger.error(e)
    result["success"] = success
    result["warn"] = warn
    return result


# BARCODE METHOD 1
def barcode_scan(
    file_path: Path, output: Path, parameters=None, debug: bool = False
) -> (bool, bool):
    success, warn = False, False
    if parameters is None:
        parameters = {}
    input_file: str = str(file_path)
    output_file: str = str(output.joinpath(file_path.name))
    # bd = cv2.barcode.BarcodeDetector()
    try:
        img = cv2.imread(input_file)
    except OSError as e:
        logger.error(e)
        return success, warn

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
    closed = cv2.dilate(closed, None, iterations=9)

    # cv2.imshow("Image cloed", closed)
    # plt.imshow(closed)
    # plt.show()

    # cnts0, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    try:
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    except IndexError:
        return success, warn
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.intp(box)
    # draw a bounding box arounded the detected barcode and display the
    # image
    if debug:
        img_d = img.copy()

        # cv2.drawContours(img_d, cnts0, -1, (255, 0, 0), 3)  # in blue
        cv2.drawContours(img_d, [box], -1, (255, 0, 0), 12)
        #
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

    (x, y), (w, h), angle = rect
    x = math.ceil(x)
    y = math.ceil(y)
    h = math.ceil(h)
    w = math.ceil(w)

    if angle > 45:
        logger.debug("rotated")
        w, h = h, w
        angle = 90 - angle

    height, width = img.shape[:2]

    logger.debug(f"box: {x=} {y=} {w=} {h=} {angle=} img: {width=} x {height=}  ")

    aspect = w / h
    aspect_ideal = 3.9386
    aspect_corrected = aspect / aspect_ideal

    corr_bar_size_width = w
    corr_bar_size_height = h * aspect_corrected * 1.02

    corr_img_size_width = width
    corr_img_size_height = height * aspect_corrected

    logger.debug(f"corr bar: {corr_bar_size_width=} x {corr_bar_size_height=}  ")
    logger.debug(f"corr img: {corr_img_size_width=} x {corr_img_size_height=}  ")

    M = cv2.moments(box)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    logger.debug(f"box: {cX=} {cY=} ")
    angle_rot = -angle  # -(90 - angle)
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

    logger.debug(f"CROP AREA: {top=} {bottom=} {left=} {rigth=}")

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

    if w < 250 or h < 60:
        logger.debug("box: not found by size ")
        return success, warn

    if aspect < 3.2:
        logger.debug("box: not found by aspect ")
        return success, warn

    try:
        cv2.imwrite(output_file, img_rotated_crop)
        success = True
    except OSError as e:
        logger.error(e)
    # leave memory
    img_rotated_crop = np.zeros(0)
    img_rotated = np.zeros(0)
    img = np.zeros(0)
    gray = np.zeros(0)
    return success, warn


# BARCODE METHOD 2
def barcode_linear_cv(
    file_path: Path, output: Path, parameters=None, debug: bool = False
) -> (bool, bool):
    # print(cv2.__version__)
    success, warn = False, False
    if parameters is None:
        parameters = {}
    input_file: str = str(file_path)
    output_file: str = str(output.joinpath(file_path.name))
    try:
        img = cv2.imread(input_file)
    except OSError as e:
        logger.error(e)
        return success, warn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bd = cv2.barcode.BarcodeDetector()
    retval, points = bd.detect(gray)
    if not retval:
        logger.debug("BOX NOT FOUND")
        return success, warn
    # print(points)
    # sort contours by area anf get bigger
    try:
        points = sorted(points, key=cv2.contourArea, reverse=True)[0]
    except IndexError:
        logger.debug("contours problems")
        return success, warn
    # print(points)

    points = scale_contour(points, 0.84, 1.15)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype(np.int32)
    # print(box)
    (x, y), (w, h), angle = rect
    x = math.ceil(x)
    y = math.ceil(y)
    h = math.ceil(h)
    w = math.ceil(w)

    if w < h:
        logger.debug("swapped : w < h")
        w, h = h, w
        angle = 90 - angle - 1
    else:
        angle = 1 - angle

    height, width = img.shape[:2]

    logger.debug(f"box: {x=} {y=} {w=} {h=}  img: {width=} x {height=} {angle=} ")
    if debug:
        img_d = img.copy()
        cv2.drawContours(img_d, [box], -1, (0, 255, 0), 12)
        # print(
        #     retval,
        #     box,
        # )
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
        plt.figure(1)
        plt_img_rows = 1
        plt_img_cols = 4
        plt_img = 1
        plt.subplot(plt_img_rows, plt_img_cols, plt_img)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt_img += 1
        plt.subplot(plt_img_rows, plt_img_cols, plt_img)
        plt.imshow(cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB))

    aspect = w / h
    aspect_ideal = 3.9386
    aspect_corrected = aspect / aspect_ideal

    corr_bar_size_width = w
    corr_bar_size_height = h * aspect_corrected * 1.02

    corr_img_size_width = width
    corr_img_size_height = height * aspect_corrected

    logger.debug(f"corr bar: {corr_bar_size_width=} x {corr_bar_size_height=}  ")
    logger.debug(f"corr img: {corr_img_size_width=} x {corr_img_size_height=}  ")

    M = cv2.moments(box)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    logger.debug(f"box: {cX=} {cY=} ")
    if w < h:
        angle_rot = 90 - angle  # -(90 - angle)
    else:
        angle_rot = -angle  # -(90 - angle)
    logger.debug(f"box: {cX=} {cY=} {angle_rot=}")
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

    logger.debug(f"CROP AREA: {top=} {bottom=} {left=} {rigth=}")

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

    if w < 250 or h < 60:
        logger.debug("box: not found by size ")
        return success, warn

    if aspect < 3.2:
        logger.debug("box: not found by aspect ")
        return success, warn

    try:
        cv2.imwrite(output_file, img_rotated_crop)
        success = True
    except OSError as e:
        logger.error(e)
    # leave memory
    img_rotated_crop = np.zeros(0)
    img_rotated = np.zeros(0)
    img = np.zeros(0)
    gray = np.zeros(0)
    return success, warn


# if __name__ == "__main__":
#     barcode_01()
#     # barcode_qr()
#     # barcode_qr_cv()
#     # barcode_linear_cv()s
