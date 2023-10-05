import multiprocessing
from datetime import datetime
from pathlib import Path

# from progressbar import progressbar
from tqdm import tqdm
import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import freeze_support
import concurrent.futures

from ai_crop_images.image_barcode import im_scan_barcode
from ai_crop_images.image_scanner import im_scan, iteration_scan
from ai_crop_images.parse_args import app_arg

from ai_crop_images.logger import (
    get_logger_multicore,
    build_queue_listener,
    worker_configurer,
)


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
    log_path = Path("log")
    log_path.mkdir(exist_ok=True, parents=True)
    LOG_FILES = {
        "warning": log_path / Path("files_warning.log"),
        "skipped": log_path / Path("files_skipped.log"),
    }

    path_out = Path(output_dir)
    if not path_out.exists():
        path_out.mkdir()

    repair_out: Path | None = repair
    barcode_method = parameters.get("barcode_method", 0)
    # Scan single image specified by command line argument --image <IMAGE_PATH>
    if im_file_path:
        im_file = Path(im_file_path)
        if im_file.suffix.lower() not in VALID_FORMATS:
            logger.debug(
                f"[bold red]File '{im_file_path}' not is {VALID_FORMATS}[/bold red]"
            )
            return

        if im_file.exists() and im_file.is_file():
            if barcode_method:
                result = im_scan_barcode(
                    im_file,
                    path_out,
                    parameters=parameters,
                    debug=debug,
                    barcode_method=barcode_method,
                )
            else:
                result = im_scan(im_file, path_out, parameters=parameters, debug=debug)
            if not result.get("success"):
                print(result)
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
        futures = []
        queue, listener = build_queue_listener()
        with concurrent.futures.ProcessPoolExecutor() as pool:
            for i in range(total_files_not_pass):
                im = im_files_not_pass[i]
                # print(f"{i}. im_scan({im})")
                if im.is_file():
                    if barcode_method:
                        future = pool.submit(
                            im_scan_barcode,
                            im,
                            path_out,
                            parameters,
                            False,
                            barcode_method,
                            queue,
                            worker_configurer,
                        )
                        futures.append(future)
                    else:
                        future = pool.submit(
                            iteration_scan,
                            im,
                            path_out,
                            parameters,
                            queue,
                            worker_configurer,
                        )
                        futures.append(future)

            results = [
                future.result()
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                )
            ]

        queue.put_nowait(None)
        listener.join()

        for result in results:
            if not result["success"]:
                skipped.append(result["im"])
            if result["warn"]:
                warning.append(result["im"])
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


def init_logger(arg_log_path: Path, debug: bool = False, log: bool = False):
    global logger
    # logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    logger = get_logger_multicore("ai_crop_images", arg_log_path, debug, log)


def init_logger_iscan(_name: str = None):
    global logger
    name = "" if _name is None else f".{_name}"
    logger = logging.getLogger(f"{__name__}{name}")


def cli():
    freeze_support()
    args = app_arg()
    # logger.setLevel(logging.DEBUG if args.debug else logging.ERROR)
    log_dir = Path("log")
    logfile = log_dir.joinpath("main_debug.log")
    # log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.ERROR,
        handlers=[RotatingFileHandler(logfile, maxBytes=20000, backupCount=10)],
    )
    init_logger(log_dir, debug=args.debug, log=True)
    # handler = RotatingFileHandler(logfile, maxBytes=20000, backupCount=10)
    # logger.setLevel(logging.DEBUG if args.debug else logging.ERROR)
    # logger.addHandler(handler)
    # for _ in range(10000):
    #     logger.debug("Hello, world!")
    # exit()
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)

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
        "barcode_method": args.barcode_method,
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


logger: logging
# logger = logging.getLogger("ai_crop_images")
# logging.basicConfig(level=logging.ERROR)

if __name__ == "__main__":
    cli()
