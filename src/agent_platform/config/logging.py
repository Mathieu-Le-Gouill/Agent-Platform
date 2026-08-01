import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler.setFormatter(formatter)

    # avoid duplicate log lines if setup_logging is called more than once (e.g. reload)
    if not root.handlers:
        root.addHandler(handler)


__all__ = ["setup_logging"]
