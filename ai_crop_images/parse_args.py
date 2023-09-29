import argparse
import sys
from pefile import PE
from pathlib import Path

if sys.version_info >= (3, 8):
    from importlib.metadata import version
else:
    from importlib_metadata import version


def get_version_PE():
    if getattr(sys, "frozen", False):
        pe = PE(sys.executable)
        if not "VS_FIXEDFILEINFO" in pe.__dict__:
            print("ERROR: Oops, has no version info. Can't continue.")
            return None
        if not pe.VS_FIXEDFILEINFO:
            print("ERROR: VS_FIXEDFILEINFO field not set for. Can't continue.")
            return None
        verinfo = pe.VS_FIXEDFILEINFO[0]
        # print(verinfo)
        filever = (
            verinfo.FileVersionMS >> 16,
            verinfo.FileVersionMS & 0xFFFF,
            verinfo.FileVersionLS >> 16,
            # verinfo.FileVersionLS & 0xFFFF,
        )
        return "{}.{}.{}".format(*filever)


def get_version():
    try:
        version_str = version(__package__)
    except Exception:
        version_str = get_version_PE()
        if version_str is None:
            version_str = "undefined"
    pack = __package__ if __package__ else Path(sys.executable).name

    return f"Version: '{ version_str }', package: {pack}"


def app_arg():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-V",
        "--version",
        action="version",
        version=get_version(),
        help="show version of app",
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="Directory of images to be scanned")
    group.add_argument("--image", help="Path to single image to be scanned")
    ap.add_argument(
        "--output",
        default="output",
        help="Directory to output result images, default: 'output'",
    )
    ap.add_argument(
        "--repair",
        default=None,
        help="If the output folder is not empty, then save to the recovery folder, by default: None",
    )
    ap.add_argument(
        "--gamma",
        type=float,
        default="4.0",
        help="Gamma image correction pre-filter, default: '4.0', 1 - Off",
    )
    ap.add_argument(
        "--morph",
        type=int,
        default="35",
        help="morph image correction for smooth contours, default: '35'. 0 - Off",
    )
    ap.add_argument(
        "--blur",
        type=int,
        choices=(3, 5, 7, 9, 11, 13),
        default=5,
        help="image blur kernel size, default: '5'",
    )
    ap.add_argument(
        "--normalize",
        default="1",
        help="normalize_scale image correction pre-filter, "
        "default: '1'. 1 - Off, 1.2 - for start",
    )
    ap.add_argument(
        "--dilate",
        action="store_true",
        help="dilate, CV operation to close open contours with an eclipse. default: 'off'",
    ),
    ap.add_argument(
        "--ratio",
        type=float,
        default="1.294",
        help="desired correction of the image aspect ratio H to W, default: '1.294'",
    )
    ap.add_argument(
        "--min_height",
        type=int,
        default="1000",
        help="desired minimum height of the output image in px, default: '1000'",
    )
    ap.add_argument(
        "--detection_height",
        type=int,
        default="900",
        help="internally downscale the original image to this height in px "
        "for the found border, default: '900'",
    )
    ap.add_argument(
        "--no_iteration",
        action="store_true",
        help="disable the iteration process to automatically adjust the gamma and dilate"
        " values in case of an unsuccessful result, default: iteration is enabled.",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="debug, CV operation for single image only",
    )
    ap.add_argument(
        "--log",
        action="store_true",
        help="store a list of skipped images and images with comments in log files",
    )
    ap.add_argument(
        "--noskip",
        action="store_true",
        help="no skip wrong images, like output same size, "
        "or result less than 800x1000. Copy original if problem. Default: skipped",
    )
    ap.add_argument(
        "--all_input",
        action="store_true",
        help="Scan all images in the input folder without skipping the search "
        "for already processed images in the output folder",
    )

    args = ap.parse_args()

    # print(args)
    return args
