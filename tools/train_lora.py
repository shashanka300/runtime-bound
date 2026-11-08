"""LoRA SFT of Gemma 4 31B (QAT W4A16, the weights the scorer serves) on the agent's own resolved
runs (tools/build_sft.py). docs/design_changes.md, 2026-11-06.

Rendering follows vLLM: the chat template is rendered to text, then encoded without special tokens
(trackw/replay.py). The loss covers only the model's own turns: for every assistant message k the
target is render(messages[:k+1]) minus render(messages[:k], add_generation_prompt=True), plus the
stop token <|tool_response> after a tool call. Logits are computed only at target positions, in
chunks (the 262k vocabulary would need ~6 GB per 12k tokens otherwise). The int4 Linears are wrapped
by tools/qlora.py (PEFT cannot wrap them); the adapter is saved in PEFT format.

    infra/train/.venv/bin/python tools/train_lora.py data/lora/sft_v1.jsonl --out data/lora/adapter_v1
    infra/train/.venv/bin/python tools/train_lora.py data/lora/sft_v1.jsonl --smoke   # 2 steps, memory
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SNAP = next((ROOT / "data/hf/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots").iterdir())
TOOLS = ROOT / "data/lora/tools.json"
STOP_AFTER_CALL = "<|tool_response>"


def to_oai(m: dict) -> dict:
    if m["role"] == "assistant" and m.get("tool_calls") is not None:
        return {"role": "assistant", "content": None, "reasoning": m.get("thinking") or "",
                "tool_calls": [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"], "arguments": c["arguments"]}}
                               for c in m["tool_calls"]]}
    if m["role"] == "tool":
        return {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
    return {"role": m["role"], "content": m.get("content") or ""}


def merge_turns(msgs: list[dict]) -> list[dict]:
    """Consecutive assistant steps (a thought-only step, then the next call or the final reply) are
    one model turn in the template; merge them so the rendering is prefix-stable."""
    out: list[dict] = []
    for m in msgs:
        if out and m["role"] == "assistant" and out[-1]["role"] == "assistant" and not out[-1].get("tool_calls"):
            p = out[-1]
            r = "\n\n".join(x for x in (p.get("reasoning"), m.get("reasoning")) if x)
            if p.get("content"):
                r = "\n\n".join(x for x in (r, p["content"]) if x)
            out[-1] = {**m, "reasoning": r}
        else:
            out.append(dict(m))
    return out


def misused(content: str) -> bool:
    """A tool call that failed as a call (bad arguments, edit target not found, write outside the
    workspace, missing file), as opposed to a command that ran and exited non-zero (a failing repro or
    test, grep with no match), which is information the agent asked for."""
    try:
        r = json.loads(content)
    except (TypeError, ValueError):
        return False
    if not isinstance(r, dict):
        return False
    if "error" in r and r.get("status") is None:  # the framework's own argument check ("Invoking `x()` failed ...")
        return True
    return r.get("status") == "error" and r.get("error_type") != "CommandError"


def mark_errors(msgs: list[dict]) -> int:
    """Error-step masking (SWE-Lego, Step Rejection Fine-Tuning): a model turn whose tool call failed
    as a call stays in the context but gets no loss, so the model learns to recover from the mistake
    without learning to make it. Sets m["_mask"]; returns the number of masked turns."""
    out = {m["tool_call_id"]: m.get("content") for m in msgs if m["role"] == "tool"}
    n = 0
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            if any(misused(out.get(c["id"])) for c in m["tool_calls"]):
                m["_mask"] = True
                n += 1
    return n


def encode(tok, msgs: list[dict], tools: list, max_len: int, loss_from: int = 0):
    """Token ids and labels (-100 outside the model's own turns from index loss_from on),
    truncated at a turn boundary. Returns (ids, labels, k_next) with k_next the index of the first
    assistant turn left out (None if all fit), or None if the rendering is not prefix-stable."""
    def text(n: int, gen: bool) -> str:
        return tok.apply_chat_template(msgs[:n], tools=tools, tokenize=False,
                                       add_generation_prompt=gen, enable_thinking=True)

    ids: list[int] = []
    labels: list[int] = []
    prev = ""
    stop_id = tok.convert_tokens_to_ids(STOP_AFTER_CALL)
    for k, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prompt = text(k, True)
        full = text(k + 1, False)
        if not prompt.startswith(prev) or not full.startswith(prompt):
            return None  # template not prefix-stable here: drop the example
        ctx = tok.encode(prompt[len(prev):], add_special_tokens=False)
        tgt = tok.encode(full[len(prompt):], add_special_tokens=False)
        if m.get("tool_calls") and not full.endswith(STOP_AFTER_CALL):
            tgt.append(stop_id)  # the template closes a call turn with it; add it if absent
            full = full + STOP_AFTER_CALL
        prev = full
        if len(ids) + len(ctx) + len(tgt) > max_len:
            return ids, labels, k
        ids += ctx + tgt
        learn = k >= loss_from and not m.get("_mask")
        labels += [-100] * len(ctx) + (tgt if learn else [-100] * len(tgt))
    return ids, labels, None


def windows(tok, msgs: list[dict], tools: list, max_len: int, overlap: int = 2) -> list | None:
    """Split a long run into examples of <= max_len tokens: system + task + the last `overlap`
    model turns (context, no loss) + new turns (loss). The host compacts long sessions too, so
    the model never sees the whole history at once either."""
    head = msgs[:2]
    asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    out = []
    start = asst[0] if asst else None  # first assistant index (in msgs) still to be learned
    ov = overlap
    while start is not None:
        pos = asst.index(start)
        ctx0 = asst[max(0, pos - ov)] if out else start
        sub = head + msgs[ctx0:]
        shift = ctx0 - len(head)
        enc = encode(tok, sub, tools, max_len, loss_from=start - shift)
        if enc is None:
            return out or None
        ids, labels, k_next = enc
        if not any(x != -100 for x in labels):
            if ctx0 != start:  # the context turns leave no room: retry without them
                ov = 0
                continue
            # this turn (with its tool output) alone exceeds max_len: skip it, keep the rest of the run
            start = asst[pos + 1] if pos + 1 < len(asst) else None
            ov = overlap
            continue
        out.append((ids, labels))
        ov = overlap
        start = None if k_next is None else k_next + shift
    return out


def chunked_loss(model, hidden: torch.Tensor, labels: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Mean next-token CE over labelled positions, logits built only there (with Gemma's softcap)."""
    pos = (labels[1:] != -100).nonzero().squeeze(-1)
    h = hidden[pos]  # hidden at t predicts token t+1
    y = labels[1:][pos]
    head = model.get_output_embeddings()
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, len(pos), chunk):
        logits = head(h[i:i + chunk]).float()
        if cap:
            logits = torch.tanh(logits / cap) * cap
        total = total + F.cross_entropy(logits, y[i:i + chunk], reduction="sum")
    return total / max(1, len(pos))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data/lora/adapter_v1")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    # all Linears: attention-only LoRA underperforms even at equal parameter count (LoRA Without Regret)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-runs", type=int, default=5, help="runs held out for a validation loss per epoch")
    ap.add_argument("--no-error-mask", action="store_true", help="also train on turns whose tool call failed")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-fraction", type=float, default=0.95)
    ap.add_argument("--cpu-offload", action="store_true", help="keep saved activations in pinned CPU RAM")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last/state.pt (saved every 10 steps)")
    ap.add_argument("--smoke", action="store_true", help="2 optimizer-free steps; report memory and speed")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    import sys

    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(ROOT / "tools"))
    from qlora import QLinear, save_adapter, wrap_model

    tok = AutoTokenizer.from_pretrained(SNAP)
    tools = json.loads(TOOLS.read_text())
    rows = [json.loads(ln) for ln in args.data.open() if ln.strip()]
    random.shuffle(rows)
    val_rows = [] if args.smoke else rows[:args.val_runs]
    rows = rows[len(val_rows):]
    masked = [0, 0]

    def build(rs: list[dict]) -> tuple[list, int]:
        out, dropped = [], 0
        for r in rs:
            msgs = merge_turns([to_oai(m) for m in r["messages"]])
            if not args.no_error_mask:
                masked[0] += mark_errors(msgs)
            masked[1] += sum(1 for m in msgs if m["role"] == "assistant")
            ws = windows(tok, msgs, tools, args.max_len)
            if ws:
                out += ws
            else:
                dropped += 1
        return out, dropped

    data, dropped = build(rows)
    val, _ = build(val_rows)
    print(f"runs dropped (template not prefix-stable): {dropped}; model turns masked (failed tool call): "
          f"{masked[0]}/{masked[1]}", flush=True)
    n_tgt = sum(sum(x != -100 for x in lab) for _, lab in data)
    print(f"examples {len(data)} from {len(rows)} runs; tokens {sum(len(i) for i, _ in data)}; target tokens {n_tgt}; "
          f"val {len(val)} windows from {len(val_rows)} runs ({[r['id'] for r in val_rows]})", flush=True)

    t0 = time.time()
    # stay inside VRAM: past it the WSL driver spills to system RAM and a step takes ~20 min (11-07)
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    print(f"loaded {type(model).__name__} in {time.time() - t0:.0f}s; "
          f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    model.config.use_cache = False
    targets = args.targets.split(",")
    lora = wrap_model(model, targets, args.rank, args.alpha)  # PEFT cannot wrap the packed int4 Linears
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"LoRA on {len(lora)} modules; trainable {sum(p.numel() for p in params) / 1e6:.1f}M; "
          f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    base = model
    with torch.no_grad():  # sanity: the model's own solved runs should already have a low loss
        ids0, lab0 = min(data, key=lambda d: len(d[0]))
        h = base.model(input_ids=torch.tensor([ids0], device="cuda")).last_hidden_state[0]
        print(f"initial loss on its own run ({len(ids0)} tokens): {chunked_loss(base, h, torch.tensor(lab0, device='cuda')).item():.4f}",
              flush=True)
        del h
    # AdamW on CPU: its state (2 x 122M fp32 = 1 GiB) on the GPU made every step after the first
    # OOM (11-08). The GPU keeps the LoRA weights and their gradients; the CPU keeps master copies.
    master = [torch.nn.Parameter(p.detach().to("cpu", copy=True)) for p in params]
    opt = torch.optim.AdamW(master, lr=args.lr, weight_decay=0.0)

    def opt_step() -> None:
        for m, p in zip(master, params):
            m.grad = None if p.grad is None else p.grad.detach().to("cpu", torch.float32)
            p.grad = None
        torch.nn.utils.clip_grad_norm_(master, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for m, p in zip(master, params):
                p.copy_(m.data)
    steps = 2 if args.smoke else math.ceil(len(data) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, 1 - s / max(1, steps)))
    import contextlib

    offload = (lambda: torch.autograd.graph.save_on_cpu(pin_memory=True)) if args.cpu_offload else contextlib.nullcontext
    def val_loss() -> float:
        """Token-weighted loss on the held-out runs (no grad)."""
        model.eval()
        tot = n = 0.0
        with torch.no_grad():
            for ids_v, lab_v in val:
                k = sum(x != -100 for x in lab_v)
                hv = base.model(input_ids=torch.tensor([ids_v], device="cuda")).last_hidden_state[0]
                tot += chunked_loss(base, hv, torch.tensor(lab_v, device="cuda")).item() * k
                n += k
                del hv
        model.train()
        return tot / max(1.0, n)

    spe = len(data) / args.accum  # optimizer steps per epoch
    saves = {min(steps, round(spe * e)): e for e in range(1, math.ceil(args.epochs) + 1)}
    if val and not args.smoke:
        print(f"val loss before training: {val_loss():.4f}", flush=True)
    order: list[int] = []

    def next_window():
        nonlocal order
        if not order:
            order = list(range(len(data)))
            random.shuffle(order)
            if args.smoke:  # the longest examples, for peak memory
                order.sort(key=lambda i: len(data[i][0]))
        return data[order.pop()]

    last = args.out / "last"  # resumable state: LoRA weights, optimizer, scheduler, step, RNG
    start = 0
    if args.resume and (last / "state.pt").exists():
        st = torch.load(last / "state.pt", weights_only=False)
        lora_mods = {n: m for n, m in model.named_modules() if isinstance(m, QLinear) and m.r}
        for n, (a, b) in st["lora"].items():
            lora_mods[n].lora_A.data.copy_(a)
            lora_mods[n].lora_B.data.copy_(b)
        with torch.no_grad():
            for m, p in zip(master, params):
                m.copy_(p.detach().cpu())
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start = st["step"]
        for _ in range(start * args.accum):  # replay the data order up to the saved step
            next_window()
        print(f"resumed at step {start}", flush=True)
    skipped = 0
    model.train()
    for step in range(start, steps):
        t1 = time.time()
        loss_acc = 0.0
        try:
            for _ in range(1 if args.smoke else args.accum):
                ids, lab = next_window()
                x = torch.tensor([ids], device="cuda")
                y = torch.tensor(lab, device="cuda")
                with offload():  # checkpointed layer inputs kept in CPU RAM (--cpu-offload)
                    out = base.model(input_ids=x)  # decoder only; hidden states, no full-vocab logits
                # loss outside the offload: inside it, every logits chunk copied the 2.6 GiB tied
                # embedding (saved for backward) to pinned RAM; that exhausted it at step 3 (11-08)
                loss = chunked_loss(base, out.last_hidden_state[0], y, chunk=256)
                if args.smoke:
                    print(f"  forward peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
                (loss / (1 if args.smoke else args.accum)).backward()
                loss_acc += loss.item()
                del out, loss, x, y
                # variable window lengths fragment the cache (no expandable_segments on WSL): 30 GiB
                # reserved with 5 GiB of it unusable crashed step 2 (11-08). Return it when the unused
                # part grows (every window doubled the step time: cudaMalloc is slow on WSL)
                if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > 2 * 2**30:
                    torch.cuda.empty_cache()
                # the pinned host cache (save_on_cpu) never shrinks and windows differ in size: it
                # grew 0 -> 8 -> 12 GiB of "shared GPU memory" in two windows (11-08); release it
                torch._C._host_emptyCache()
        except torch.OutOfMemoryError:
            oom = True
        else:
            oom = False
        if oom:  # (outside the except block, so the traceback no longer holds the tensors)
            # skip this optimizer step (its gradients are partial) instead of losing the run
            skipped += 1
            for p in params:
                p.grad = None
            out = loss = x = y = None
            torch.cuda.empty_cache()
            sched.step()
            print(f"step {step + 1}/{steps} SKIPPED (OOM; {skipped} so far)", flush=True)
            if skipped > 10:
                raise RuntimeError("more than 10 optimizer steps lost to OOM")
            continue
        opt_step()
        sched.step()
        print(f"step {step + 1}/{steps} loss {loss_acc / (1 if args.smoke else args.accum):.4f} "
              f"{time.time() - t1:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
              f"len {len(ids)}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        if not args.smoke and ((step + 1) % 10 == 0 or step + 1 == steps):
            last.mkdir(parents=True, exist_ok=True)
            torch.save({"lora": {n: (m.lora_A.data, m.lora_B.data) for n, m in model.named_modules()
                                 if isinstance(m, QLinear) and m.r},
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1},
                       last / "state.tmp")
            (last / "state.tmp").replace(last / "state.pt")
        if not args.smoke and step + 1 in saves:  # one adapter per epoch: both get screened
            e = saves[step + 1]
            d = args.out / f"epoch{e}"
            save_adapter(model, d, targets, args.rank, args.alpha, "google/gemma-4-31B-it-qat-w4a16-ct")
            print(f"saved {d}" + (f"; val loss {val_loss():.4f}" if val else ""), flush=True)


if __name__ == "__main__":
    main()
