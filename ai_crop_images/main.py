import time
import datetime
import argparse
from pathlib import Path
from ai_crop_images.image_scanner import im_scan


def print_datetime(func):
    def wrapper(*args, **kwargs):
        d = datetime.datetime.now()
        print(d)
        func(*args, **kwargs)

    return wrapper


def scan_file_dir(im_file_path: str = None, im_dir: str = None):
    valid_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]

    ##get_ext = lambda f: os.path.splitext(f)[1].lower()

    # Scan single image specified by command line argument --image <IMAGE_PATH>
    if im_file_path:
        # scanner.scan(im_file_path)
        # print(f"im_scan({im_file_path})")
        im_scan(im_file_path)
        ...

    # Scan all valid images in directory specified by command line argument --images <IMAGE_DIR>
    else:
        path = Path(im_dir)
        im_files = path.glob("*.*")

        im_files = [f for f in path.glob("*.*") if f.suffix.lower() in valid_formats]
        for im in im_files:
            # print(f"im_scan({im})")
            im_scan(im)


def app_arg():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="Directory of images to be scanned")
    group.add_argument("--image", help="Path to single image to be scanned")
    ap.add_argument(
        "-i",
        action="store_true",
        help="Flag for manually verifying and/or setting document corners",
    )
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
    scan_file_dir(args.image, args.images)
    d = datetime.datetime.now()
    print(d)
    print(f"{__package__} : cli, delay 3 sec")
    time.sleep(5)  # Sleep for 3 seconds


if __name__ == "__main__":
    cli()
