"""Tiny filesystem helpers shared by immutable artifact writers."""

from __future__ import annotations

from collections.abc import Callable
import logging
import os
from pathlib import Path
import time


def rename_directory_with_retry(
    source: Path,
    destination: Path,
    *,
    rename: Callable[[Path, Path], None] = os.rename,
    logger: logging.Logger | None = None,
    attempts: int = 7,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Commit a directory while tolerating short Windows/OneDrive locks."""

    for attempt in range(1, attempts + 1):
        try:
            rename(source, destination)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            delay = 0.25 * (2 ** (attempt - 1))
            if logger is not None:
                logger.warning(
                    "Directory rename was temporarily blocked; retrying in "
                    "%.2f seconds (%d/%d)",
                    delay,
                    attempt,
                    attempts,
                )
            sleep(delay)
