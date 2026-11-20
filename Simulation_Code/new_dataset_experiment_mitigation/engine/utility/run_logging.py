"""Attempt logs outside immutable scientific outputs, including startup failures."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
import faulthandler
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
import traceback


logger = logging.getLogger(__name__)


class _ProgressLog(logging.FileHandler):
    def __init__(self, path: Path) -> None:
        super().__init__(path, mode="x", encoding="utf-8")
        self.last_activity = time.monotonic()

    def emit(self, record: logging.LogRecord) -> None:
        self.last_activity = time.monotonic()
        super().emit(record)


def _write_record(directory: Path, name: str, **values: object) -> None:
    with (directory / name).open("x", encoding="utf-8") as stream:
        json.dump(
            {"time_utc": datetime.now(timezone.utc).isoformat(), **values},
            stream,
            ensure_ascii=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def log_run_attempt(
    output_root: Path, run_id: str, *, stall_seconds: float = 600.0
) -> Generator[Path, None, None]:
    """Record one CLI attempt without reusing a previous attempt's files.

    The watchdog only reports inactivity and dumps stacks; it never interrupts
    work or retries requests. A missing terminal record means an unclean stop,
    not a diagnosed cause. Process-tree termination/power loss cannot be caught.
    """
    if stall_seconds <= 0:
        raise ValueError("stall_seconds must be positive")
    logs_root = Path(output_root) / "_logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=logs_root))
    handler = _ProgressLog(attempt / "runtime.log")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    stop = threading.Event()
    previously_enabled = faulthandler.is_enabled()
    watchdog = None
    try:
        with (attempt / "native-crash.log").open("xb") as native_log:
            faulthandler.enable(file=native_log, all_threads=True)

            def report_inactivity() -> None:
                while not stop.wait(min(60.0, stall_seconds)):
                    silent_seconds = time.monotonic() - handler.last_activity
                    if silent_seconds >= stall_seconds:
                        logger.warning(
                            "No log progress for %.1f seconds; process is alive. "
                            "Writing thread stacks to %s (work is not interrupted)",
                            silent_seconds, native_log.name,
                        )
                        faulthandler.dump_traceback(file=native_log, all_threads=True)

            try:
                _write_record(
                    attempt, "started.json", run_id=run_id,
                    pid=os.getpid(), parent_pid=os.getppid(), status="started",
                )
                watchdog = threading.Thread(
                    target=report_inactivity, name="run-log-watchdog", daemon=True
                )
                watchdog.start()
                logger.info(
                    "Run attempt started run=%s pid=%s log_directory=%s",
                    run_id, os.getpid(), attempt,
                )
                yield attempt
                _write_record(attempt, "completed.json", run_id=run_id, status="complete")
                logger.info("Run attempt completed run=%s", run_id)
            except BaseException as exc:
                logger.exception("Run attempt failed run=%s", run_id)
                try:
                    _write_record(
                        attempt, "failed.json", run_id=run_id, status="failed",
                        exception_type=type(exc).__name__, error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                except Exception:
                    logger.exception("Could not save attempt failure record: %s", attempt)
                raise
            finally:
                stop.set()
                if watchdog is not None:
                    watchdog.join()
                faulthandler.disable()
                if previously_enabled:
                    faulthandler.enable()
    finally:
        root_logger.removeHandler(handler)
        handler.close()
