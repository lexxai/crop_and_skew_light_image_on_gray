import multiprocessing
from datetime import datetime
from pathlib import Path
from rich import print

# from random import randrange
# from time import sleep

from imutils.perspective import four_point_transform
import imutils
import cv2
from pathlib import Path
import shutil
import types
import logging

# import os
import numpy as np


logger: logging = None


def init_logger(_name: str = None):
    global logger
    name = "" if _name is None else f".{_name}"
    logger = logging.getLogger(f"{__name__}{name}")


def start_datetime(func):
    def wrapper(*args, **kwargs):
        d = datetime.now()
        logger.debug(f" *** Start:  {d}")
        return func(*args, **kwargs)

    return wrapper


def end_datetime(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        d = datetime.now()
        logger.debug(f" *** End:  {d}")
        return result

    return wrapper


def dur_datetime(func):
    def wrapper(*args, **kwargs):
        d1 = datetime.now()
        d1_fmt = d1.strftime("%Y-%m-%d %H:%M")
        logger.debug(f"*** Start:  {d1_fmt}")
        result = func(*args, **kwargs)
        d2 = datetime.now()
        diff = d2 - d1
        d2_fmt = d2.strftime("%Y-%m-%d %H:%M")
        logger.debug(f" *** End:  {d2_fmt}, duration: {diff}")
        return result

    return wrapper


def copy_original(input_file: str, output_file: str) -> bool:
    logger.debug(
        f" Problem read file '{input_file}', then just copy original to destianton "
    )
    return shutil.copyfile(input_file, output_file)


# GAMMA CODE : https://stackoverflow.com/questions/26912869/color-levels-in-opencv
def cv_gamma(image, gamma: float = 4.0):
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


def cv_scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def cv_processing(
    img_file: Path, output: Path, parameters=None, debug: bool = False
) -> tuple[bool, bool]:
    """cv_processing of image with saving file if all ok.

    Args:
        img_file (Path): Path to input single file
        output (Path): Path to saved folder
        parameters (dict): Dict of parameters. Defaults to {},
        debug (bool): flash debug CV diagnostic. Defaults to False

    Returns:
        bool: result of operation, True file saved
        bool: warning if was problem with file but it skipped
    """

    if parameters is None:
        parameters = {}
    input_file: str = str(img_file)
    output_file: str = str(output.joinpath(img_file.name))
    warning: bool = False

    green_color = (0, 255, 0)

    image_ratio: float = float(parameters.get("ratio", 1.294))
    image_gamma: float = float(parameters.get("gamma", 4.0))
    image_morph: int = int(parameters.get("morph", 35))
    image_normalize_scale: float = float(parameters.get("normalize_scale", 1.0))
    image_skip_wrong = parameters.get("skip_wrong", False)
    image_height_for_detection = int(parameters.get("detection_height", 900))
    image_dilate = parameters.get("dilate", False)
    image_geometry_ratio = image_ratio
    image_blur = int(parameters.get("blur", 5))

    MIN_HEIGHT: int = int(parameters.get("min_height", 1000))
    MIN_WIDTH: int = round(MIN_HEIGHT / image_ratio)

    #################################################################
    # Load the Image
    #################################################################
    try:
        image = cv2.imread(input_file)
    except Exception as e:
        logger.debug(f"CV INPUT ERRROR {e}")
        if not image_skip_wrong:
            copy_original(input_file, output_file)
            return True, True
        return False, True

    if isinstance(image, types.NoneType):
        logger.debug(
            f"CV INPUT ERRROR read '{input_file}' image == None. skip: {image_skip_wrong}"
        )
        if not image_skip_wrong:
            copy_original(input_file, output_file)
            return True, True
        return False, True

    orig_image = image.copy()

    image = imutils.resize(image, height=image_height_for_detection)
    # image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    if image_gamma != 1.0:
        image = cv_gamma(image, image_gamma)

    if image_normalize_scale != 1:
        image = cv_normalize_scale(image, image_normalize_scale)

    #################################################################
    # Image Processing
    #################################################################

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert the image to gray scale
    blur = cv2.GaussianBlur(gray, (image_blur, image_blur), 0)  # Add Gaussian blur

    MORPH = image_morph

    if MORPH > 0:
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

    if image_dilate:
        # it help fill gap on contours, but at result upscale contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        dilated = cv2.dilate(edged, kernel)
        contours, _ = cv2.findContours(
            dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # downscale dilated contours by 1%
        contours = [cv_scale_contour(c, 0.99) for c in contours]
        dilated = np.empty(0)
        kernel = np.empty(0)
    else:
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if debug:
        if contours:
            c_len = len(contours)
            c_step = round(255 / c_len - 1)
            image_bound = image.copy()
            for i, c in enumerate(contours):
                color = 255 - (i * c_step)
                # logger.debug(color)
                # get bounding rect
                (x, y, w, h) = cv2.boundingRect(c)
                # draw red rect
                cv2.rectangle(
                    image_bound, (x, y), (x + w, y + h), (255 - color, 0, color), 4
                )
            cv2.imshow("Image boundingRect", imutils.resize(image_bound, height=500))

        # Show the image and all the contours
        cv2.imshow("Image", imutils.resize(image, height=500))
        cv2.drawContours(image, contours, -1, green_color, 3)
        cv2.imshow("All contours", imutils.resize(image, height=500))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        image_bound = np.empty(0)

    #################################################################
    # Select Only the Edges of the Document
    #################################################################

    # go through each contour
    contours_found = False
    for contour in contours:
        # we approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
        # if we found a countour with 4 points we break the for loop
        # (we can assume that we have found our document)
        if len(approx) == 4:
            doc_cnts = approx
            contours_found = True
            break

    #################################################################
    # Apply Warp Perspective to Get the Top-Down View of the Document
    #################################################################
    coef_y = orig_image.shape[0] / image.shape[0]
    coef_x = orig_image.shape[1] / image.shape[1]

    if contours_found:
        try:
            for contour in doc_cnts:
                contour[:, 0] = contour[:, 0] * coef_y
                contour[:, 1] = contour[:, 1] * coef_x
        except UnboundLocalError:
            warning = True
            logger.debug(
                "** UnboundLocalError NOT FOUND contours, "
                "try to change gamma parameter."
            )
            if not image_skip_wrong:
                copy_original(input_file, output_file)
                return True, True
            logger.debug("Image SKIPPED.")
            return False, warning
    else:
        warning = True
        logger.debug("*******  NOT FOUND contours, " "try to change gamma parameter.")
        if not image_skip_wrong:
            copy_original(input_file, output_file)
            return True, True
        logger.debug("Image SKIPPED.")
        return False, warning

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
        # cv2.imwrite("output_" + "/" + os.path.basename(img_file), warped)
        cv2.imshow("Scanned", imutils.resize(warped, height=750))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    logger.debug(
        f"Original image dimension: {orig_image.shape[1]} x {orig_image.shape[0]} "
    )
    logger.debug(f"Result   image dimension: {warped.shape[1]} x {warped.shape[0]} ")

    if warped.shape[:2] == orig_image.shape[:2]:
        warning = True
        logger.debug(
            "******   Result is same as ORIGINAL" " try to change gamma parameter."
        )
        if image_skip_wrong:
            logger.debug("Image SKIPPED.")
            return False, warning

    h, w = warped.shape[:2]
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        warning = True
        logger.debug(
            f"******   Result is less ( {MIN_WIDTH} x {MIN_HEIGHT} )"
            " try to change gamma parameter."
        )
        if image_skip_wrong:
            logger.debug("Image SKIPPED.")
            return False, warning

    if isinstance(warped, types.NoneType):
        logger.debug(
            f"CV warped ERRROR'{input_file}' warped == None. skip: {image_skip_wrong}"
        )
        if not image_skip_wrong:
            copy_original(input_file, output_file)
            return True, True
        return False, True

    result = cv2.imwrite(output_file, warped)

    # delete all unused
    try:
        warped = np.empty(0)
        orig_image = np.empty(0)
        blur = np.empty(0)
        kernel = np.empty(0)
        image = np.empty(0)
        for contour in contours:
            contour = np.empty(0)
    except Exception as e:
        logger.debug("Clear memory", e)

    return result, warning


# @dur_datetime
def im_scan(
    file_path: Path,
    output: Path,
    parameters=None,
    debug: bool = False,
):
    # logger.debug(f"STILL FAKE. Just print :) {__package__}, im_scan {file_path}")
    if parameters is None:
        parameters = {}
    if logger is None:
        init_logger(file_path.name)
    size = file_path.stat().st_size
    date_m = datetime.fromtimestamp(file_path.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M"
    )
    modified = str(date_m)
    logger.debug(f"File: '{file_path.name}' {size=} bytes, {modified=}")
    return cv_processing(file_path, output, parameters=parameters, debug=debug)


def tune_parameter_dilate(parameter, id: int = None) -> tuple[dict, int]:
    logger.debug(" --- Automatically add 'dilate' option as last way")
    parameter_copy = parameter.copy()
    parameter_copy["dilate"] = True
    # logger.debug(parameter_copy)
    return parameter_copy, id + 1


def tune_parameter_gamma(parameter, id: int = None) -> tuple[dict, int]:
    """_summary_

    Args:
        parameter (dict): parameters dict
        id (int): id of steps

    Returns:
        tuple[dict, float]: copy parameter , id of next steps
    """
    STEPS = (0, -1, -1.5, -2, -2.5, -3, -3.5, 1, 1.5, 2, 2.5, 3, 3.5, 4)
    # STEPS = (0,)

    gamma_start = parameter["gamma"]
    if id is not None:
        if id < len(STEPS):
            parameter_copy = parameter.copy()
            while True:
                if id >= len(STEPS):
                    return tune_parameter_dilate(parameter, id)
                    # return None, None
                step = STEPS[id]
                gamma = gamma_start + step
                logger.debug(
                    f"tune_parameter_gamma id={id+1}, {step=}, {gamma=} {gamma>1}"
                )
                if gamma > 1:
                    parameter_copy["gamma"] = gamma
                    return parameter_copy, id + 1
                else:
                    id += 1

        else:
            if id == len(STEPS):
                return tune_parameter_dilate(parameter, id)
            else:
                return None, None

    return parameter, id


def iteration_scan(
    im: Path,
    path_out: Path,
    parameters: dict,
    queue: multiprocessing.Queue = None,
    configurer=None,
) -> dict:
    result = {"success": None, "warn": None, "im": im}
    success = None
    warn = None
    if configurer is not None:
        configurer(queue)
    init_logger(im.name)
    # init_logger_iscan(im.name)
    if parameters.get("no_iteration", False):
        success, warn = im_scan(
            im,
            path_out,
            parameters=parameters,
        )
    else:
        is_done = False
        iteration = 0
        while not is_done:
            parameters_work, iteration = tune_parameter_gamma(parameters, iteration)
            if parameters_work is not None:
                gamma = parameters_work["gamma"]
                dilate = parameters_work["dilate"]
                logger.debug(f"# {iteration=}, {gamma=}, {dilate=}")
                success, warn = im_scan(
                    im,
                    path_out,
                    parameters=parameters_work,
                )
                if not warn:
                    is_done = True
            else:
                logger.debug(" ***** All iterations failed, operation failed")
                break
    result["success"] = success
    result["warn"] = warn
    return result
