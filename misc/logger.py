import logging
import os
import sys
from contextlib import contextmanager

logger_format = '%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s'
logging.basicConfig(format=logger_format, level=logging.INFO)


def get_logger(name):
    return logging.getLogger(name)


class DuplicateFilter:
    """
    Filters away duplicate log messages.
    Modified version of: https://stackoverflow.com/a/31953563/965332
    """

    def __init__(self, logger):
        self.msgs = set()
        self.logger = logger

    def filter(self, record):
        msg = str(record.msg)
        is_duplicate = msg in self.msgs
        if not is_duplicate:
            self.msgs.add(msg)
        return not is_duplicate

    def __enter__(self):
        if self not in self.logger.filters:
            self.logger.addFilter(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


logger = get_logger("task_logger")
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(logger_format)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)


@contextmanager
def task_file_log_scope(task_name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    debug_log_path = os.path.join(log_dir, f"{task_name}_debug.log")

    handler = logging.FileHandler(debug_log_path, encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(logger_format)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    try:
        yield debug_log_path
    finally:
        handler.close()
        logger.removeHandler(handler)