from datetime import datetime
from pathlib import Path
from rich import print
from random import randrange
from time import sleep


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


@dur_datetime
def im_scan(file_path: Path):
    print(f"STILL FAKE. Just print :) {__package__}, im_scan {file_path}")
    size = file_path.stat().st_size
    modified = str(datetime.fromtimestamp(file_path.stat().st_mtime))
    print(f"{size=} bytes, {modified=}")
    sleep(randrange(5, 40) / 10.0)
