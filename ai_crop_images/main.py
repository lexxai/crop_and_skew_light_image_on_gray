from datetime import datetime
from pathlib import Path

# from progressbar import progressbar
from tqdm import tqdm
import logging

from ai_crop_images.image_barcode import im_scan_barcode
from ai_crop_images.image_scanner import im_scan
from ai_crop_images.parse_args import app_arg


import sys
import gc

from rich import print


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
        d = datetime.now()
        print(d)
        func(*args, **kwargs)

    return wrapper


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
            f.write(
                "# Datetime: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M"))
            )
            f.write("# App args: {}\n".format(" ".join(sys.argv[1:])))
            f.writelines([f"{d}\n" for d in data])


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

    repair_out: Path | None = repair
    barcode_base = parameters.get("barcode_base", False)

    # Scan single image specified by command line argument --image <IMAGE_PATH>
    if im_file_path:
        im_file = Path(im_file_path)
        if im_file.suffix.lower() not in VALID_FORMATS:
            print(f"[bold red]File '{im_file_path}' not is {VALID_FORMATS}[/bold red]")
            return

        if im_file.exists() and im_file.is_file():
            if barcode_base:
                im_scan_barcode(im_file, path_out, parameters=parameters, debug=debug)
            else:
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

        # skip search same files on output_ folder
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
            if repair_out:
                repair_out.mkdir(exist_ok=True, parents=True)
                path_out = repair_out

        print(
            f"total input files: {total_files}, ready for operations: {total_files_not_pass}"
        )

        skipped = []
        warning = []
        for i in tqdm(range(total_files_not_pass), total=total_files_not_pass):
            im = im_files_not_pass[i]
            # print(f"{i}. im_scan({im})")
            if im.is_file():
                if barcode_base:
                    success, warn = im_scan_barcode(im, path_out, parameters=parameters)
                else:
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


def cli():
    args = app_arg()
    # logger.setLevel(logging.DEBUG if args.debug else logging.ERROR)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.ERROR)
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
        "barcode_base": args.blur,
    }
    try:
        scan_file_dir(
            args.output,
            args.image,
            args.images,
            parameters=parameters,
            debug=args.debug,
            log=args.log,
            repair=args.repair,
        )
    except Exception as e:
        print(f"Other error of 'scan_file_dir': {e}")
    d = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\nEND: {d}")


logger = logging.getLogger()
# logging.basicConfig(level=logging.ERROR)

if __name__ == "__main__":
    cli()
