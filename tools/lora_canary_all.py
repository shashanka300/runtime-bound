"""All-Linear LoRA canary for the trained-adapter layout (tools/qlora.py save_adapter):
`make`: rank-16 adapter on q/k/v/o/gate/up/down of every layer, shapes from the checkpoint's
weight_shape tensors. Attention lora_B = 0, MLP lora_B random, so the output changes only if vLLM
applies the MLP LoRA. `compare`: base vs canary at temperature 0 (text, top-5 logprobs), plus decode
speed of each (256 tokens).
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/probes/lora_canary_all"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
R, ALPHA = 16, 32


def make():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    snap = next((ROOT / "data/hf/hub/models--google--gemma-4-31B-it-qat-w4a16-ct/snapshots").iterdir())
    g = torch.Generator().manual_seed(0)
    sd = {}
    with safe_open(str(snap / "model.safetensors"), "pt") as h:
        for k in h.keys():
            if not (k.endswith(".weight_shape") and "language_model" in k):
                continue
            mod = k[: -len(".weight_shape")]
            if mod.rsplit(".", 1)[1] not in TARGETS:
                continue
            o, i = (int(x) for x in h.get_tensor(k).tolist())
            a = torch.randn(R, i, generator=g) / i ** 0.5
            mlp = ".mlp." in mod
            b = torch.randn(o, R, generator=g) * (0.02 if mlp else 0.0)
            sd[f"base_model.model.{mod}.lora_A.weight"] = a.to(torch.bfloat16)
            sd[f"base_model.model.{mod}.lora_B.weight"] = b.to(torch.bfloat16)
    OUT.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(OUT / "adapter_model.safetensors"))
    (OUT / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "google/gemma-4-31b-it-qat-w4a16-ct", "r": R, "lora_alpha": ALPHA,
        "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False, "target_modules": TARGETS,
        "inference_mode": True}, indent=1))
    print(OUT, len(sd) // 2, "modules")


def compare():
    import httpx
    models = [m["id"] for m in httpx.get("http://127.0.0.1:8000/v1/models").json()["data"]]
    base = next(m for m in models if "gemma" in m.lower())
    res = {}
    for model in (base, "canary_all"):
        r = httpx.post("http://127.0.0.1:8000/v1/chat/completions", json={
            "model": model, "messages": [{"role": "user", "content": "Write one sentence about the ocean."}],
            "max_tokens": 24, "temperature": 0.0, "logprobs": True, "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False}}, timeout=600).json()
        c = r["choices"][0]
        t0 = time.time()
        s = httpx.post("http://127.0.0.1:8000/v1/completions", json={
            "model": model, "prompt": "def fibonacci(n):", "max_tokens": 256, "min_tokens": 256,
            "temperature": 0.0, "ignore_eos": True}, timeout=600).json()
        dt = time.time() - t0
        res[model] = {"text": c["message"]["content"],
                      "top5": [(t["token"], round(t["logprob"], 3)) for t in c["logprobs"]["content"][0]["top_logprobs"]],
                      "decode_tok_s": round(s["usage"]["completion_tokens"] / dt, 1)}
    res["mlp_lora_applied"] = (res[base]["text"] != res["canary_all"]["text"]
                               or res[base]["top5"] != res["canary_all"]["top5"])
    (ROOT / "data/probes/lora_canary_all.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    make() if sys.argv[1] == "make" else compare()
