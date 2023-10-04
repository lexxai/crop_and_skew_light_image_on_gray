import logging
from pathlib import Path

import logging.handlers
import multiprocessing


log_base_path: Path | None = None
# logged_format = "%(asctime)s [%(levelname)s] pid:%(process)s (%(name)s.%(funcName)s:%(lineno)d) %(message)s"
logged_format = "%(asctime)s [%(levelname)s] #%(process)s (%(name)s) %(message)s"
logged_format_date = "%Y-%m-%d %H:%M:%S"
logged_formatter = logging.Formatter(logged_format, datefmt=logged_format_date)


def get_logger_multicore(
    name: str, log_path: Path | None = None, debug: bool = False, log: bool = False
) -> logging:
    global log_base_path
    my_logger: logging = logging.getLogger(name)
    if log:
        log_path: Path = Path() if log_path is None else log_path
        log_path.mkdir(exist_ok=True, parents=True)
        log_base_path = log_path
    return my_logger


def get_logger(name: str, log_path: Path | None = None, debug: bool = False) -> logging:
    global log_base_path
    log_path: Path = Path() if log_path is None else log_path
    log_path.mkdir(exist_ok=True, parents=True)
    log_base_path = log_path
    # create file handler for DEBUG, INFO
    file_name = "debug.log"
    file_path = log_path.joinpath(file_name)
    logged_handler_file = logging.FileHandler(str(file_path))
    logged_handler_file.setLevel(logging.DEBUG)
    logged_handler_file.setFormatter(logged_formatter)
    # create file handler for ERRORS
    file_name = "error.log"
    file_path = log_path.joinpath(file_name)
    logged_handler_error_file = logging.FileHandler(str(file_path))
    logged_handler_error_file.setLevel(logging.ERROR)
    logged_handler_error_file.setFormatter(logged_formatter)
    # create console handler INFO, ERROR
    logged_handler_stream = logging.StreamHandler()
    logged_handler_stream.setLevel(logging.INFO)
    logged_handler_stream.setFormatter(logged_formatter)
    # create main logger for all child modules of thi package
    my_logger: logging = logging.getLogger(name)
    my_logger.addHandler(logged_handler_file)
    my_logger.addHandler(logged_handler_error_file)
    my_logger.addHandler(logged_handler_stream)

    my_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    return my_logger


# Because you'll want to define the logging configurations for listener and workers, the
# listener and worker process functions take a configurer parameter which is a callable
# for configuring logging for that process. These functions are also passed the queue,
# which they use for communication.
#
# In practice, you can configure the listener however you want, but note that in this
# simple example, the listener does not apply level or filter logic to received records.
# In practice, you would probably want to do this logic in the worker processes, to avoid
# sending events which would be filtered out between processes.
#
# The size of the rotated files is made small so you can see the results easily.
def listener_configurer(log_base: Path = None):
    root = logging.getLogger()
    if log_base is None:
        # log_base = Path("logs")
        # log_base.mkdir(exist_ok=True, parents=True)
        root.addHandler(logging.NullHandler())
        return
    root = logging.getLogger()
    log_max_size = 3000000
    log_max_count = 10
    # INFO, ERROR SAVE TO ROTATED FILE
    log_path = log_base.joinpath("error.log")
    h_errors = logging.handlers.RotatingFileHandler(
        log_path, "a", log_max_size, log_max_count
    )
    h_errors.setLevel(logging.ERROR)
    h_errors.setFormatter(logged_formatter)
    # INFO SAVE TO ROTATED FILE
    log_path = log_base.joinpath("debug.log")
    h_debug = logging.handlers.RotatingFileHandler(
        log_path, "a", log_max_size, log_max_count
    )
    h_debug.setLevel(logging.DEBUG)
    h_debug.setFormatter(logged_formatter)

    # create console handler INFO, ERROR
    log_path = log_base.joinpath("info.log")
    h_info = logging.handlers.RotatingFileHandler(
        log_path, "a", log_max_size, log_max_count
    )
    h_info.setLevel(logging.INFO)
    h_info.setFormatter(logged_formatter)

    root.addHandler(h_info)
    root.addHandler(h_debug)
    root.addHandler(h_errors)
    # root.addHandler(logged_handler_stream)


# This is the listener process top-level loop: wait for logging events
# (LogRecords)on the queue and handle them, quit when you get a None for a
# LogRecord.
def listener_process(queue: multiprocessing.Queue, configurer, log_base: Path = None):
    configurer(log_base)
    while True:
        try:
            record = queue.get()
            if (
                record is None
            ):  # We send this as a sentinel to tell the listener to quit.
                break
            # print(record)
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback

            print("Whoops! Problem:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            break


# The worker configuration is done at the start of the worker process run.
# Note that on Windows you can't rely on fork semantics, so each process
# will run the logging configuration code when it starts.
def worker_configurer(queue: multiprocessing.Queue):
    root = logging.getLogger()
    if not root.hasHandlers():
        h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
        root.addHandler(h)
        root.setLevel(logging.DEBUG)
    # send all messages, for demo; no other level or filter logic applied.


def build_queue_listener(
    log_base: Path = None,
) -> (multiprocessing.Queue, multiprocessing.Process):
    if log_base is None:
        log_base = log_base_path
    queue = multiprocessing.Manager().Queue(-1)
    listener = multiprocessing.Process(
        target=listener_process, args=(queue, listener_configurer, log_base)
    )
    listener.start()
    return queue, listener


# if __name__ == "__main__":
#     main()
