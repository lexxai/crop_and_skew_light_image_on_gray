from time import sleep
import datetime
import argparse
from pathlib import Path
from progressbar import progressbar
from ai_crop_images.image_scanner import im_scan
from rich import print


def exception_keyboard(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt:
            print("EXIT !!!")
            exit()

    return wrapper


def print_datetime(func):
    def wrapper(*args, **kwargs):
        d = datetime.datetime.now()
        print(d)
        func(*args, **kwargs)

    return wrapper


# for i in progressbar(range(100), redirect_stdout=True):
#     print("Some text", i)
#     sleep(0.1)


@exception_keyboard
def scan_file_dir(
    output_dir: str,
    im_file_path: str = None,
    im_dir: str = None,
):
    VALID_FORMATS = (".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif")

    # Scan single image specified by command line argument --image <IMAGE_PATH>
    if im_file_path:
        im_scan(im_file_path)
        ...

    # Scan all valid images in directory specified by command line argument --images <IMAGE_DIR>
    else:
        path_in = Path(im_dir)
        im_files = path_in.glob("*.*")

        im_files = [f for f in path_in.glob("*.*") if f.suffix.lower() in VALID_FORMATS]

        path_out = Path(output_dir)
        output_files = path_in.glob("*.*")
        output_files = [
            f for f in path_out.glob("*.*") if f.suffix.lower() in VALID_FORMATS
        ]

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

        total_files = len(im_files)
        total_files_not_pass = len(im_files_not_pass)
        print(
            f"total input files: {total_files}, ready for operations: {total_files_not_pass}"
        )
        for i in progressbar(range(total_files_not_pass), redirect_stdout=True):
            im = im_files_not_pass[i]
            # print(f"{i}. im_scan({im})")
            im_scan(im)
            # sleep(2)


def app_arg():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="Directory of images to be scanned")
    group.add_argument("--image", help="Path to single image to be scanned")
    ap.add_argument(
        "--output",
        default="output",
        help="Path to output result images, default: 'output'",
    )
    # args = vars(ap.parse_args())
    args = ap.parse_args()

    print(args)
    # im_dir = args.images
    # im_file_path = args.image
    # interactive_mode = args.i
    return args


def cli():
    args = app_arg()
    scan_file_dir(args.output, args.image, args.images)
    d = datetime.datetime.now()
    print(d)
    print(f"{__package__} : cli, delay 3 sec")
    sleep(5)  # Sleep for 3 seconds


if __name__ == "__main__":
    cli()
