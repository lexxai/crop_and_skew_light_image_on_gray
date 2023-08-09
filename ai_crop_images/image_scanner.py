from datetime import datetime
from pathlib import Path


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


@end_datetime
@start_datetime
def im_scan(file_path: Path):
    print(f"{__package__}, im_scan {file_path}")
    size = file_path.stat().st_size
    modified = str(datetime.fromtimestamp(file_path.stat().st_mtime))
    print(f"{size=} bytes, {modified=}")
