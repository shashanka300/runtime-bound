"""LoRA v1 training on a Modal H100 (80 GB): same trainer, data and settings as the local run
(tools/train_lora.py; docs/design_changes.md 2026-11-08), without CPU offload.

One-time setup (user):  pip install modal && modal token new
Run (from the repo root):
    modal run tools/modal_train.py::upload             # local model snapshot (19 GB) -> volume, once
    modal run --detach tools/modal_train.py::main      # train; adapters land in the volume
    modal run tools/modal_train.py::fetch              # copy adapters to data/lora/adapter_v1_modal
"""

import subprocess
import threading
import time
from pathlib import Path

import modal

app = modal.App("gemma4-lora-v1")
vol = modal.Volume.from_name("gemma4-lora", create_if_missing=True)
REPO = "/root/challeenge"
MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.13.1", "safetensors==0.8.0", "compressed-tensors==0.15.0.1",
                 "accelerate")
    .add_local_file("tools/train_lora.py", f"{REPO}/tools/train_lora.py")
    .add_local_file("tools/qlora.py", f"{REPO}/tools/qlora.py")
    .add_local_file("data/lora/sft_v2.jsonl", f"{REPO}/data/lora/sft_v2.jsonl")
    .add_local_file("data/lora/tools.json", f"{REPO}/data/lora/tools.json")
)


SNAP_REL = "hf/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots"


@app.local_entrypoint()
def upload():
    """Our local copy of the checkpoint (the exact files the local trainer and vLLM use) into the
    volume, in the HF cache layout the trainer expects. Symlinked blobs are uploaded as files."""
    local = next(Path(f"data/{SNAP_REL}").iterdir())
    files = sorted(p for p in local.iterdir() if p.is_file())
    with vol.batch_upload(force=True) as batch:
        for p in files:
            batch.put_file(str(p.resolve()), f"/{SNAP_REL}/{local.name}/{p.name}")
    print("uploaded", [p.name for p in files])


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=24 * 3600)
def train(epochs: float = 2.0, max_len: int = 8192):
    # the trainer reads the model from <repo>/data/hf/hub (HF cache layout), kept in the volume
    assert any(Path(f"/vol/{SNAP_REL}").glob("*/model.safetensors")), "run ::upload first"
    link = Path(f"{REPO}/data/hf")
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to("/vol/hf")

    stop = threading.Event()

    def committer():  # persist checkpoints (state.pt every 10 steps, epoch adapters) as they appear
        while not stop.wait(600):
            vol.commit()

    threading.Thread(target=committer, daemon=True).start()
    cmd = ["python", "tools/train_lora.py", "data/lora/sft_v2.jsonl", "--max-len", str(max_len),
           "--epochs", str(epochs), "--targets", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
           "--out", "/vol/adapter_v1", "--resume"]
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=REPO)
    stop.set()
    vol.commit()
    print(f"exit {rc} after {(time.time() - t0) / 3600:.2f} h")
    return rc


@app.local_entrypoint()
def main(epochs: float = 2.0):
    print("exit code", train.remote(epochs=epochs))


@app.local_entrypoint()
def start(epochs: float = 2.0):
    """Fire-and-forget (use with --detach): the run no longer depends on this client's connection.
    A blocking .remote() was cancelled at step 112 when the local client hit a network timeout."""
    call = train.spawn(epochs=epochs)
    print("spawned", call.object_id)


@app.function(volumes={"/vol": vol})
def _read(path: str) -> bytes:
    return Path(path).read_bytes()


@app.function(volumes={"/vol": vol})
def _list() -> list[str]:
    return [str(p) for p in Path("/vol/adapter_v1").rglob("*") if p.is_file() and "last" not in p.parts]


@app.local_entrypoint()
def fetch():
    for p in _list.remote():
        dest = Path("data/lora/adapter_v1_modal") / Path(p).relative_to("/vol/adapter_v1")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_read.remote(p))
        print("fetched", dest)
