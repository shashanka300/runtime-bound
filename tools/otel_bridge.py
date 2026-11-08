"""Training telemetry -> OTLP (Aspire dashboard on localhost:4318), from the logs this machine already
has: the streamed Modal training log, the local training log, and the Modal credit guard log.
No change to the running jobs; nothing leaves the machine.

    ~/otel-venv/bin/python tools/otel_bridge.py
"""

import re
import time
from pathlib import Path

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

LOGS = Path(__file__).resolve().parents[1] / "data" / "logs"
SOURCES = {  # file -> run label
    LOGS / "modal_train.out": "lora_v1_modal_h100",
    LOGS / "lora_v1_train.out": "lora_v1_local_5090",
}
GUARD = LOGS / "modal_credit_guard.out"
ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\]8;[^\x1b]*\x1b\\")
STEP = re.compile(r"^step (\d+)/(\d+) loss ([0-9.]+) (\d+)s peak ([0-9.]+) GiB len (\d+)")
SKIP = re.compile(r"^step (\d+)/(\d+) SKIPPED")
VAL0 = re.compile(r"^val loss before training: ([0-9.]+)")
VAL = re.compile(r"^saved \S+/epoch(\d+); val loss ([0-9.]+)")
SPEND = re.compile(r"spend ~\$([0-9.]+)|metered \$([0-9.]+)")

provider = MeterProvider(
    resource=Resource.create({"service.name": "gemma4-lora-training"}),
    metric_readers=[PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint="http://127.0.0.1:4318/v1/metrics"), export_interval_millis=15000)],
)
metrics.set_meter_provider(provider)
m = metrics.get_meter("train")
g_loss = m.create_gauge("train.loss", description="mean loss of the optimizer step (8 windows)")
g_step = m.create_gauge("train.step", description="optimizer steps done")
g_prog = m.create_gauge("train.progress", unit="%", description="steps done / total")
g_sec = m.create_gauge("train.step_seconds", unit="s")
g_peak = m.create_gauge("train.peak_gpu_memory", unit="GiB")
g_len = m.create_gauge("train.window_tokens", unit="tokens", description="length of the step's last window")
g_val = m.create_gauge("train.val_loss", description="held-out validation loss (5 runs)")
c_skip = m.create_counter("train.skipped_steps", description="optimizer steps skipped on OOM")
g_spend = m.create_gauge("modal.spend", unit="USD", description="Modal spend (credit guard estimate)")


def follow(path: Path, pos: int) -> tuple[list[str], int]:
    if not path.exists():
        return [], 0
    size = path.stat().st_size
    if size < pos:  # file was rewritten (relaunch)
        pos = 0
    with path.open("rb") as f:
        f.seek(pos)
        data = f.read()
    end = data.rfind(b"\n") + 1
    text = ANSI.sub("", data[:end].decode("utf-8", "replace")).replace("\r", "\n")
    return text.splitlines(), pos + end


pos = {p: 0 for p in [*SOURCES, GUARD]}
while True:
    for path, run in SOURCES.items():
        lines, pos[path] = follow(path, pos[path])
        a = {"run": run}
        for ln in lines:
            ln = ln.strip()
            if s := STEP.match(ln):
                step, total = int(s[1]), int(s[2])
                g_step.set(step, a)
                g_prog.set(round(100 * step / total, 1), a)
                g_loss.set(float(s[3]), a)
                g_sec.set(int(s[4]), a)
                g_peak.set(float(s[5]), a)
                g_len.set(int(s[6]), a)
            elif s := SKIP.match(ln):
                c_skip.add(1, a)
            elif s := VAL0.match(ln):
                g_val.set(float(s[1]), {**a, "epoch": "0"})
            elif s := VAL.match(ln):
                g_val.set(float(s[2]), {**a, "epoch": s[1]})
    lines, pos[GUARD] = follow(GUARD, pos[GUARD])
    for ln in lines:
        if s := SPEND.search(ln):
            g_spend.set(float(s[1] or s[2]), {"run": "lora_v1_modal_h100"})
    time.sleep(10)
