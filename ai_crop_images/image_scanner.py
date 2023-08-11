from datetime import datetime
from pathlib import Path
from rich import print

# from random import randrange
# from time import sleep

from imutils.perspective import four_point_transform
import imutils
import cv2
from pathlib import Path

# import os
import numpy as np


def start_datetime(func):
    def wrapper(*args, **kwargs):
        d = datetime.now()
        print(f" *** Start:  {d}")
        return func(*args, **kwargs)

    return wrapper


def end_datetime(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        d = datetime.now()
        print(f" *** End:  {d}")
        return result

    return wrapper


def dur_datetime(func):
    def wrapper(*args, **kwargs):
        d1 = datetime.now()
        print(f"\n *** Start:  {d1}")
        result = func(*args, **kwargs)
        d2 = datetime.now()
        diff = d2 - d1
        print(f" *** End:  {d2}, duration: {diff}")
        return result

    return wrapper


# GAMMA CODE : https://stackoverflow.com/questions/26912869/color-levels-in-opencv
def cv_gamma(image, gamma: float = 7.0):
    inBlack = np.array([0, 0, 0], dtype=np.float32)
    inWhite = np.array([255, 255, 255], dtype=np.float32)
    inGamma = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    outBlack = np.array([0, 0, 0], dtype=np.float32)
    outWhite = np.array([255, 255, 255], dtype=np.float32)
    inGamma = inGamma / gamma
    img_g = image.copy()
    img_g = np.clip((img_g - inBlack) / (inWhite - inBlack), 0, 255)
    img_g = (img_g ** (1 / inGamma)) * (outWhite - outBlack) + outBlack
    image = np.clip(img_g, 0, 255).astype(np.uint8)
    return image


# https://stackoverflow.com/questions/42257173/contrast-stretching-in-python-opencv
def cv_normalize_scale(image, beta: float = 1.2):
    # normalize float versions
    norm_img = cv2.normalize(
        image, None, alpha=0, beta=beta, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    # scale to uint8
    norm_img = np.clip(norm_img, 0, 1)
    norm_img = (255 * norm_img).astype(np.uint8)
    return norm_img


def cv_processing(
    img_file: Path,
    output: Path,
    parameters: dict = {},
    debug: bool = False,
):
    MIN_WIDTH: int = 300
    MIN_HEIGHT: int = 300

    #################################################################
    # Load the Image
    #################################################################
    input_file: str = str(img_file)
    output_file: str = str(output.joinpath(img_file.name))

    green_color = (0, 255, 0)

    image_ratio: float = float(parameters.get("ratio", 1.294))
    image_gamma: float = float(parameters.get("gamma", 7.0))
    image_morph: int = int(parameters.get("morph", 35))

    image_geometry_ratio = image_ratio
    image_height_for_detection = 500

    image = cv2.imread(input_file)

    orig_image = image.copy()

    image = imutils.resize(image, height=image_height_for_detection)
    # image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    image = cv_gamma(image, image_gamma)
    image = cv_normalize_scale(image)

    #################################################################
    # Image Processing
    #################################################################

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert the image to gray scale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Add Gaussian blur

    MORPH = image_morph

    # dilate helps to remove potential holes between edge segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
    blur = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)

    edged = cv2.Canny(blur, 75, 200)  # Apply the Canny algorithm to find the edges

    # Show the image and the edges
    if debug:
        cv2.imshow("Original image:", imutils.resize(image, height=500))
        cv2.imshow("Edged:", imutils.resize(edged, height=500))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    # exit()

    #################################################################
    # Use the Edges to Find all the Contours
    #################################################################

    # If you are using OpenCV v3, v4-pre, or v4-alpha
    # cv.findContours returns a tuple with 3 element instead of 2
    # where the `contours` is the second one
    # In the version OpenCV v2.4, v4-beta, and v4-official
    # the function returns a tuple with 2 element
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if debug:
        # Show the image and all the contours
        cv2.imshow("Image", imutils.resize(image, height=500))
        cv2.drawContours(image, contours, -1, green_color, 3)
        cv2.imshow("All contours", imutils.resize(image, height=500))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    #################################################################
    # Select Only the Edges of the Document
    #################################################################

    # go through each contour
    for contour in contours:
        # we approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
        # if we found a countour with 4 points we break the for loop
        # (we can assume that we have found our document)
        if len(approx) == 4:
            doc_cnts = approx
            break

    #################################################################
    # Apply Warp Perspective to Get the Top-Down View of the Document
    #################################################################
    coef_y = orig_image.shape[0] / image.shape[0]
    coef_x = orig_image.shape[1] / image.shape[1]

    try:
        for contour in doc_cnts:
            contour[:, 0] = contour[:, 0] * coef_y
            contour[:, 1] = contour[:, 1] * coef_x
    except UnboundLocalError:
        print(
            "[bold red]*******  NOT FOUND contours[/bold red], "
            "try to change gamma parameter. [bold yellow]Image SKIPPED.[/bold yellow]"
        )
        return False

    # We draw the contours on the original image not the modified one

    orig_image_c = cv2.drawContours(orig_image.copy(), [doc_cnts], -1, green_color, 30)

    if debug:
        cv2.imshow("Contours of the document", imutils.resize(orig_image_c, height=500))
        # apply warp perspective to get the top-down view

    warped = four_point_transform(orig_image, doc_cnts.reshape(4, 2))

    w = int(warped.shape[1])
    h = int(warped.shape[0])

    h = int(image_geometry_ratio * w)

    warped = cv2.resize(warped, (w, h))

    # convert the warped image to grayscale
    # warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    if debug:
        # cv2.imwrite("output" + "/" + os.path.basename(img_file), warped)
        cv2.imshow("Scanned", imutils.resize(warped, height=750))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    print(f"Original image dimension: {orig_image.shape[1]} x {orig_image.shape[0]} ")
    print(f"Result   image dimension: {warped.shape[1]} x {warped.shape[0]} ")

    if warped.shape[:2] == orig_image.shape[:2]:
        print(
            "[bold red]******   Result is same as ORIGINAL[/bold red]"
            " try to change gamma parameter. [bold yellow]Image SKIPPED.[/bold yellow]"
        )
        return False

    w, h = warped.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        print(
            f"[bold red]******   Result is less ( {MIN_WIDTH} x {MIN_HEIGHT} )[/bold red]"
            " try to change gamma parameter. [bold yellow]Image SKIPPED.[/bold yellow]"
        )
        return False

    result = cv2.imwrite(output_file, warped)
    return result


@dur_datetime
def im_scan(file_path: Path, output: Path, parameters: dict = {}, debug: bool = False):
    # print(f"STILL FAKE. Just print :) {__package__}, im_scan {file_path}")
    size = file_path.stat().st_size
    date_m = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%x %X")
    modified = str(date_m)
    print(f"File: '{file_path.name}' {size=} bytes, {modified=}")
    return cv_processing(file_path, output, parameters=parameters, debug=debug)
