"""Read-only engine binding evidence for the benchmark's report boundaries.

The two harnesses use one shared reference model. Record its existing engine
table once, rather than attributing every engine to every implementation label.
Keys and options distinguish engines; identical-entry aliases share one record.
This is a process-local binding snapshot, not per-call kernel-use evidence.
No engine is created, no binding/self-check hook is called, and no tensor is read.
"""
from __future__ import annotations

import dataclasses
import json


def _dataclass_fields(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    raise TypeError(f"Binding metadata contains unsupported {type(value).__name__}")


def snapshot_model_bindings(model):
    """Detach plain JSON diagnostics; never let a snapshot error mask a run failure.

    ``captured`` describes successful serialization only. ``direct=None`` means
    that engine has not bound yet; missing dense fields or a missing engine table
    do not establish activation. Errors remain explicit, without stringifying
    unsupported tensor/engine objects or retaining them in the result.
    """
    report = {"scope": "existing engines on this process's shared benchmark model; report boundary, not per-call dispatch",
              "status": "not_created", "engines": []}
    try:
        # Same model-owned table as engine._ModelEngineTables. Reading __dict__
        # avoids importing inference modules or invoking an engine factory.
        table = vars(model).get("_qwen38_solution_engines")
        if table is None:
            return report
        report["status"] = "captured"
        for index, (key, engine) in enumerate(table.items()):
            record = {"index": index, "class": type(engine).__name__}
            try:
                state = vars(engine)
                payload = {"key": key, "options": state["options"], "direct": state.get("direct")}
                if "capture_io" in state:
                    payload["capture_io"] = state["capture_io"]
                record.update(json.loads(json.dumps(payload, default=_dataclass_fields, allow_nan=False)))
                record["status"] = "captured"
            except Exception as error:
                record.update(status="error", error=f"{type(error).__name__}: {error}"[:500])
                report["status"] = "error"
            report["engines"].append(record)
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}"[:500])
    return report
