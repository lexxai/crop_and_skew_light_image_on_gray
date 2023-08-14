# from time import sleep
import datetime
import argparse
from pathlib import Path
from progressbar import progressbar
from ai_crop_images.image_scanner import im_scan
import sys
import os
from pefile import PE
import gc

from rich import print

if sys.version_info >= (3, 8):
    from importlib.metadata import version
else:
    from importlib_metadata import version


def exception_keyboard(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt:
            print("EXIT")
            exit()
        # except Exception as e:
        #     print(f"[bold red]ERROR: {e}[/bold red]")

    return wrapper


def print_datetime(func):
    def wrapper(*args, **kwargs):
        d = datetime.datetime.now()
        print(d)
        func(*args, **kwargs)

    return wrapper


def get_version_PE():
    if getattr(sys, "frozen", False):
        pe = PE(sys.executable)
        if not "VS_FIXEDFILEINFO" in pe.__dict__:
            print("ERROR: Oops, {} has no version info. Can't continue.").format(pename)
            return None
        if not pe.VS_FIXEDFILEINFO:
            print(
                "ERROR: VS_FIXEDFILEINFO field not set for {}. Can't continue."
            ).format(pename)
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


def tune_parameter_dilate(parameter, id: int = None) -> tuple[dict, int]:
    print("[yellow] --- Automatically add 'dilate' option as last way[/yellow]")
    parameter_copy = parameter.copy()
    parameter_copy["dilate"] = True
    # print(parameter_copy)
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
                print(f"tune_parameter_gamma id={id+1}, {step=}, {gamma=} {gamma>1}")
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


def iteration_scan(im: Path, parameters: dict, path_out: Path) -> tuple[bool, bool]:
    success = None
    warn = None
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
                print(f"\n[green]# {iteration=}, {gamma=}, {dilate=}[/green]")
                success, warn = im_scan(
                    im,
                    path_out,
                    parameters=parameters_work,
                )
                if not warn:
                    is_done = True
            else:
                print("\n[red] ***** All iterations failed, operation failed[/red]\n")
                break
    return success, warn


def save_log_file(log_file: Path, data: list[str]) -> None:
    if data:
        with open(str(log_file), "w") as f:
            for d in data:
                f.write(f"{d}\n")


@exception_keyboard
def scan_file_dir(
    output_dir: str,
    im_file_path: str = None,
    im_dir: str = None,
    parameters: dict = {},
    debug: bool = False,
    log: bool = False,
    repair: str = None,
):
    VALID_FORMATS = (".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif")
    LOG_FILES = {"warning": Path("warning.log"), "skipped": Path("skipped.log")}

    path_out = Path(output_dir)
    if not path_out.exists():
        path_out.mkdir()

    repair_out = Path(repair)

    # Scan single image specified by command line argument --image <IMAGE_PATH>
    if im_file_path:
        im_file = Path(im_file_path)
        if im_file.suffix.lower() not in VALID_FORMATS:
            print(f"[bold red]File '{im_file_path}' not is {VALID_FORMATS}[/bold red]")
            return

        if im_file.exists() and im_file.is_file():
            im_scan(im_file, path_out, parameters=parameters, debug=debug)
        else:
            print(f"[bold red]File '{im_file_path}' not found[/bold red]")
            return

    # Scan all valid images in directory specified by command line argument --images <IMAGE_DIR>
    else:
        path_in = Path(im_dir)
        if not path_in.exists():
            print(f"[bold red]Folder '{im_dir}' not found[/bold red]")
            return

        im_files = path_in.glob("*.*")

        im_files = list(
            filter(lambda f: f.suffix.lower() in VALID_FORMATS, path_in.glob("*.*"))
        )

        # skip search same files on output folder
        if not parameters.get("all_input", False):
            output_files = path_in.glob("*.*")

            output_files = list(
                filter(
                    lambda f: f.suffix.lower() in VALID_FORMATS, path_out.glob("*.*")
                )
            )

            im_files_not_pass = []
            for i in im_files:
                is_found = False
                for ind, o in enumerate(output_files):
                    if i.name == o.name:
                        output_files.pop(ind)
                        is_found = True
                        break
                if not is_found:
                    im_files_not_pass.append(i)

            im_files_not_pass = sorted(list(im_files_not_pass))
        else:
            im_files_not_pass = im_files

        total_files = len(im_files)
        total_files_not_pass = len(im_files_not_pass)

        if total_files != total_files_not_pass:
            if not repair_out.is_dir():
                repair_out.mkdir()
            path_out = repair_out

        print(
            f"total input files: {total_files}, ready for operations: {total_files_not_pass}"
        )
        skipped = []
        warning = []
        for i in progressbar(range(total_files_not_pass), redirect_stdout=True):
            im = im_files_not_pass[i]
            # print(f"{i}. im_scan({im})")
            if im.is_file():
                success, warn = iteration_scan(im, parameters, path_out)
                if not success:
                    skipped.append(im)
                if warn:
                    warning.append(im)
            # be ready for new loop
            gc.collect()

        if skipped:
            skipped_total = len(skipped)
            print(f"[yellow]Total SKIPPED files: {skipped_total}[/yellow]")
            print("\n".join([f.name for f in skipped]))
        if warning:
            warning_total = len(warning)
            print(f"[yellow]Total WARNING files: {warning_total}[/yellow]")
            print("\n".join([f.name for f in warning]))
        if log:
            save_log_file(LOG_FILES["skipped"], skipped)
            save_log_file(LOG_FILES["warning"], warning)


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


def cli():
    args = app_arg()

    parameters = {
        "gamma": float(args.gamma),
        "min_height ": int(args.min_height),
        "ratio": float(args.ratio),
        "morph": int(args.morph),
        "dilate": args.dilate,
        "normalize_scale": float(args.normalize),
        "skip_wrong": not args.noskip,
        "detection_height": int(args.detection_height),
        "all_input": args.all_input,
        "no_iteration": args.no_iteration,
        "blur": args.blur,
    }
    scan_file_dir(
        args.output,
        args.image,
        args.images,
        parameters=parameters,
        debug=args.debug,
        log=args.log,
        repair=args.repair,
    )
    d = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\nEND: {d}")


if __name__ == "__main__":
    cli()
