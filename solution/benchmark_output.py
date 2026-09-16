"""Preserve earlier benchmark reports before a new attempt can fail at startup.

Only files owned by the benchmark writers and their trace tree are moved. Raw
reports, compressed reports and trace-relative paths remain together beneath
``previous/<unique attempt>/``. Reader outputs and unrelated files stay put.
No output is removed, and this helper performs no inference or timing.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import json
from pathlib import Path
import tempfile


REPORT_PATHS = ("status.json", "bench.json", "bench.json.gz", "summary.md",
                "validation.json", "failure.json", "traces")


@contextmanager
def benchmark_startup(output):
    """Archive earlier reports, mark this attempt running, record startup errors.

    Call after argument validation and the dry-run return, before model loading.
    The caller's existing runtime/report handling takes over after this context.
    A process killed after startup can leave ``running``; exit status and current
    completion records are still required. The archive is retention, not resume.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    previous = None
    paths = [output / name for name in REPORT_PATHS if (output / name).exists()]
    if paths:
        history = output / "previous"
        history.mkdir(exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S-")
        archive = Path(tempfile.mkdtemp(prefix=stamp, dir=history))
        # mkdtemp returns an absolute path (Python 3.12) while the job shells pass
        # relative output directories; name the archive by its own components.
        previous = f"previous/{archive.name}"
        # Retire the old completion marker first. A failed move cannot leave it
        # labeling a partly archived attempt as the current successful run.
        for path in paths:
            path.rename(archive / path.name)
    status = {"status": "running", "previous_attempt": previous}
    (output / "status.json").write_text(json.dumps(status) + "\n")
    try:
        yield previous
    except BaseException as error:
        status.update(status="failed", phase="startup", error=repr(error)[:2000])
        (output / "status.json").write_text(json.dumps(status) + "\n")
        raise
