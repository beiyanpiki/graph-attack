import logging
import os
import types


def new_critical(self, msg, *args, **kwargs):
    if self.isEnabledFor(logging.CRITICAL):
        self._log(logging.CRITICAL, msg, args, **kwargs)
        self._log(logging.CRITICAL, "Fatal error. Exit 1.", args, **kwargs)
    logging.shutdown()
    exit(1)


def create_logger(log_path: str, logger_name: str = "default") -> logging.Logger:
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger: logging.Logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    filer = logging.FileHandler(log_path, mode="w")
    filer.setLevel(logging.INFO)
    streamer = logging.StreamHandler()
    # streamer.setLevel(logging.DEBUG)
    streamer.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        f"[%(asctime)s][%(levelname)1.1s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    filer.setFormatter(formatter)
    streamer.setFormatter(formatter)
    logger.addHandler(filer)
    logger.addHandler(streamer)

    # Override fatal method, if fatal error occurred, exit(1)
    logger.critical = types.MethodType(new_critical, logger)
    logger.fatal = logger.critical

    return logger
