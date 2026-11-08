import polars as pl
import pytest
from src import frontier as fr
from src.variants import make_variant


def _row(res: list[bool], sub: list[bool], chars: list[int], secs: list[float]) -> dict:
    return {"resolved": res, "submitted": sub, "patch_chars": chars, "agent_seconds": secs}


def test_deploy_stops_at_first_submission() -> None:
    r = _row([False, True, True], [False, True, True], [0, 50, 60], [100.0, 80.0, 90.0])
    s = fr.score_k(r, 3)
    assert s["deploy"] is True and s["attempts_run"] == 2
    assert s["deploy_seconds"] == pytest.approx(180.0)
    assert s["oracle"] is True and s["oracle_seconds"] == pytest.approx(270.0)


def test_deploy_without_submission_uses_last_patch() -> None:
    r = _row([False, True], [False, False], [10, 0], [50.0, 60.0])
    s = fr.score_k(r, 2)
    assert s["deploy"] is False  # attempt 0 holds the last non-empty patch
    assert s["oracle"] is True and s["attempts_run"] == 2


def test_pareto_and_h1() -> None:
    rows = []
    for i in range(20):
        repo = "a" if i < 10 else "b"
        # large model k=1 resolves tasks 0-9; small model k=2 resolves 0-9 too, slower
        rows.append(
            {
                "model": "L",
                "thinking": 4096,
                "instance_id": f"t{i}",
                "repo": repo,
                "k": 1,
                "deploy": i < 10,
                "deploy_seconds": 100.0,
                "oracle": i < 10,
                "oracle_seconds": 100.0,
                "attempts_run": 1,
            }
        )
        rows.append(
            {
                "model": "S",
                "thinking": 4096,
                "instance_id": f"t{i}",
                "repo": repo,
                "k": 2,
                "deploy": i < 10,
                "deploy_seconds": 150.0,
                "oracle": i < 10,
                "oracle_seconds": 150.0,
                "attempts_run": 2,
            }
        )
    tasks = pl.DataFrame(rows)
    cfg = fr.configs(tasks)
    assert fr.pareto(cfg).height == 1  # S is slower and not better
    verdict, table = fr.h1_test(tasks, cfg, "L", "S")
    assert verdict == "SUPPORTED" and table.height == 1  # equal rate at the 150 s budget


def test_variant_rewrites_model_thinking_and_adapters(tmp_path) -> None:
    base = tmp_path / "base"
    (base / "configs").mkdir(parents=True)
    (base / "sub_agents").mkdir()
    (base / "adapters" / "main_lora").mkdir(parents=True)
    (base / "agent.yaml").write_text(
        "name: a\nmodel: gemma-4-31b-it-qat-w4a16-ct\nadapter: main_lora\ninstruction: hi\n"
    )
    (base / "sub_agents" / "s.yaml").write_text("name: s\nmodel: x\nadapter: tool_lora\n")
    (base / "configs" / "sampling.yaml").write_text(
        "temperature: 0.2\nmax_output_tokens: 16384\nthinking_config:\n  thinking_budget: 4096\n"
    )
    out = make_variant(base, tmp_path / "v", "gemma-4-12b-it", 2048, 4096)
    agent = (out / "agent.yaml").read_text()
    assert "model: gemma-4-12b-it" in agent and "adapter" not in agent
    assert "model: gemma-4-12b-it" in (out / "sub_agents" / "s.yaml").read_text()
    assert not (out / "adapters").exists()
    sampling = (out / "configs" / "sampling.yaml").read_text()
    assert "max_output_tokens: 6144" in sampling and "thinking_budget: 2048" in sampling


def test_paired_comparison() -> None:
    rows = []
    for i in range(30):
        for k, ok in ((1, i < 10), (4, i < 20)):
            rows.append(
                {
                    "model": "M",
                    "thinking": 4096,
                    "instance_id": f"t{i}",
                    "repo": "a" if i % 2 else "b",
                    "k": k,
                    "deploy": ok,
                    "deploy_seconds": 100.0 * k,
                    "oracle": ok,
                    "oracle_seconds": 100.0 * k,
                    "attempts_run": k,
                }
            )
    r = fr.paired(
        pl.DataFrame(rows),
        {"model": "M", "thinking": 4096, "k": 1},
        {"model": "M", "thinking": 4096, "k": 4},
    )
    assert r["diff"] == pytest.approx(10 / 30) and r["significant"]
    assert (r["only_a"], r["only_b"]) == (0, 10)
