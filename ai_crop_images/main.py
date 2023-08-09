import time
import datetime


def cli():
    d = datetime.datetime.now()
    print(d)
    print(f"{__package__} : cli, delay 3 sec")
    time.sleep(5)  # Sleep for 3 seconds


if __name__ == "__main__":
    cli()
