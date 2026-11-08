"""Build and validate leaderboard submissions (declarative, < 3 GiB, zip-rooted).

    uv run python leaderboard/build.py lb01_reserve_fix

Every submission is generated from code here (never edited by hand), validated with the
swebox compiler and the competition's file rules, and zipped to
``$DATA_ROOT/submissions/<name>/submission.zip`` for upload (Kaggle requires that file name). ``leaderboard/README.md`` logs what
was submitted, the local evidence behind it and the score it received.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path

import yaml

from rbcore.paths import competition_dir, data_root
from swebox.submission import ServingTarget, compile_submission

ROOT = Path(__file__).parent
ALLOWED = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
MAX_BYTES = 3 * 1024**3
TOOLS = [
    "run_command",
    "read_file",
    "edit_file",
    "write_file",
    "get_status",
    "submit_patch",
    "get_code_neighbors",
    "search_similar_code",
    "get_code_subgraph",
]

# Per-task budgets read by the official harness. The starter ships max_time_minutes: 1 and
# max_tool_calls: 10, far below anything tested locally. 120 tasks x 5 min = 10 h < 12 h.
EVAL_BUDGET = {
    "evaluation": {
        "timeout_seconds": 300,
        "max_tool_calls": 500,
        "max_time_minutes": 5,
        "max_turns": 500,
    }
}


def lb01_reserve_fix(out: Path) -> None:
    """Starter + max_output_tokens = thinking + 4096 (E1b arm B: 18.8% -> 31.2% on dev40,
    paired +12.5 [+5.0, +21.3]) + realistic per-task budgets."""
    shutil.copytree(competition_dir() / "sample_submission", out)
    sampling = out / "configs" / "sampling.yaml"
    cfg = yaml.safe_load(sampling.read_text())
    cfg["max_output_tokens"] = int(cfg["thinking_config"]["thinking_budget"]) + 4096
    sampling.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "eval_config.yaml").write_text(
        "# Per-task budgets (see leaderboard/build.py)\n" + yaml.safe_dump(EVAL_BUDGET)
    )


# lb01 scored 0.03 (2/58). The scorer's patched vLLM 0.19.1 wipes LoRA weights (forum 743508),
# so adapters only add risk (the official sample with adapters FAILED); and timeout_seconds 300
# let one hanging test command consume the whole 5 min task budget.
EVAL_BUDGET_V2 = {
    "evaluation": {
        "timeout_seconds": 60,
        "max_tool_calls": 500,
        "max_time_minutes": 5,
        "max_turns": 500,
    }
}


def drop_adapters(out: Path) -> None:
    """Remove adapters/ and every ``adapter:`` line (YAML with !include tags: edit as text)."""
    shutil.rmtree(out / "adapters", ignore_errors=True)
    for y in [*out.rglob("*.yaml"), *out.rglob("*.yml")]:
        lines = y.read_text().splitlines(keepends=True)
        kept = [ln for ln in lines if not ln.lstrip().startswith("adapter:")]
        if kept != lines:
            y.write_text("".join(kept))


def lb02_no_lora_t60(out: Path) -> None:
    """lb01 without LoRA adapters (Kaggle's vLLM wipes them) and with a 60 s command timeout."""
    lb01_reserve_fix(out)
    drop_adapters(out)
    (out / "eval_config.yaml").write_text(
        "# Per-task budgets (see leaderboard/build.py)\n" + yaml.safe_dump(EVAL_BUDGET_V2)
    )


def _set_sampling(out: Path, **updates: object) -> None:
    sampling = out / "configs" / "sampling.yaml"
    cfg = yaml.safe_load(sampling.read_text())
    cfg.update(updates)
    sampling.write_text(yaml.safe_dump(cfg, sort_keys=False))


def lb03_thinking_off(out: Path) -> None:
    """lb02 with thinking off and max_output_tokens 4096 (E1b arm G). The transfer-gap ladder
    showed the scorer's engine (vLLM 0.19.1) makes each thinking step ~2x longer, so tasks time
    out (arm F there: 17.5% vs 30% on vLLM 0.30); the scorer ignores thinking_budget."""
    lb02_no_lora_t60(out)
    _set_sampling(out, max_output_tokens=4096, thinking_config={"include_thoughts": False})


def lb03h_think_cap4k(out: Path) -> None:
    """lb02 with thinking kept on but bounded the only way the scorer honours:
    max_output_tokens 4096 (E1b arm H)."""
    lb02_no_lora_t60(out)
    _set_sampling(out, max_output_tokens=4096)


PUBLIC_RECIPE = ROOT / "_public_recipe_honghanhhh_024"
"""Public bundle (Apache-2.0 Kaggle notebooks): honghanhhh/gemma-4-baseline-lb-0-24, itself from
flexonafft/gemma-4-budget-aware-coding-agent-v2 (lineage credited in leaderboard/README.md).
Files extracted byte for byte from kawh1111/g4-baseline-024, a config-identical fork."""


def lb04_public_7min(out: Path) -> None:
    """The strongest public recipe on the fixed scorer (31B, thinking on, budget 4096,
    max_output 8192, 28 calls, 240 s commands, call-plan prompt) with the per-task cap lowered
    from 8 to 7 minutes: the 8-minute bytes have both scored and exceeded the 12 h run limit
    (forum notebook 'One Bundle, Three Outcomes'); 7 minutes buys ~40 min of margin."""
    shutil.copytree(PUBLIC_RECIPE, out)
    ev = out / "eval_config.yaml"
    ev.write_text(ev.read_text().replace("max_time_minutes: 8", "max_time_minutes: 7"))
    prompt = out / "prompts" / "system.md"
    text = prompt.read_text()
    assert text.count("28 tool calls and 8 minutes") == 1
    prompt.write_text(text.replace("28 tool calls and 8 minutes", "28 tool calls and 7 minutes"))


def lb05a_think2048(out: Path) -> None:
    """lb04 with thinking_budget 2048: on the V31 scorer the budget reaches vLLM
    (thinking_token_budget), so a smaller budget means shorter steps and more turns."""
    lb04_public_7min(out)
    sampling = out / "configs" / "sampling.yaml"
    text = sampling.read_text()
    assert text.count("thinking_budget: 4096") == 1
    sampling.write_text(text.replace("thinking_budget: 4096", "thinking_budget: 2048"))


AUDIT_RULES = (
    "- Never restate, summarise or re-quote a file you have just read; act on it.\n"
    "- `read_file` returns at most 150 lines: always pass `start_line` and `end_line`. For files "
    "longer than about 400 lines, locate the lines with `git grep -n` first.\n"
    "- Keep reasoning short. If you have deliberated for a long time without a tool call, make "
    "the next concrete tool call now.\n"
)


def lb05b_audit_rules(out: Path) -> None:
    """lb04 plus three rules from public runtime audits (forum 755291): no restating files,
    ranged reads under the 150-line cap, short reasoning between tool calls."""
    lb04_public_7min(out)
    prompt = out / "prompts" / "system.md"
    text = prompt.read_text()
    anchor = "- Never look outside /workspace."
    assert text.count(anchor) == 1
    prompt.write_text(text.replace(anchor, AUDIT_RULES + anchor))


def lb06_12b(out: Path) -> None:
    """lb04 on gemma-4-12b-it-qat-w4a16-ct (added to the scorer 2026-11): faster decode means
    more turns inside the same 7-minute cap."""
    lb04_public_7min(out)
    agent = out / "agent.yaml"
    text = agent.read_text()
    assert text.count("model: gemma-4-31b-it-qat-w4a16-ct") == 1
    agent.write_text(text.replace("gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct"))


LB07_RULES = (
    "- `read_file` returns at most 150 lines: always pass `start_line` and `end_line`. For long "
    "files, locate the lines with `git grep -n` first.\n"
    "- In `edit_file`, never put a triple-quoted docstring line in `old_string`; anchor on a "
    "nearby unique code line instead. If an edit_file call is rejected for missing arguments, "
    "do not resend it: switch to the inline Python heredoc replacement described in the editing section below.\n"
)


def lb07_safe_tools(out: Path) -> None:
    """lb04 with the forum's evidence applied (docs/archive/IDEAS_NOV5.md): graph tools removed from the
    tool list (no measured benefit; get_code_subgraph takes a list, and vLLM 0.19.1's gemma4
    parser can loop forever on array args, freezing the scorer, forum 753973); 36 calls / 6 min
    (14/39 lb04 tasks hit the 28-call cap; the time cap drops to keep the 12 h total);
    temperature 1.0 (forum 746250); ranged-read and docstring-safe edit rules."""
    lb04_public_7min(out)
    agent = out / "agent.yaml"
    text = agent.read_text()
    for t in ("get_code_neighbors", "search_similar_code", "get_code_subgraph"):
        assert text.count(f"  - {t}\n") == 1
        text = text.replace(f"  - {t}\n", "")
    agent.write_text(text)
    ev = out / "eval_config.yaml"
    t = ev.read_text()
    assert "max_tool_calls: 28" in t and "max_time_minutes: 7" in t
    ev.write_text(t.replace("max_tool_calls: 28", "max_tool_calls: 36")
                  .replace("max_time_minutes: 7", "max_time_minutes: 6"))
    prompt = out / "prompts" / "system.md"
    text = prompt.read_text()
    graph = ("- Never use `search_similar_code`, `get_code_neighbors` or `get_code_subgraph`. "
             "They return nothing useful.\n")
    assert text.count(graph) == 1 and text.count("28 tool calls and 7 minutes") == 1
    text = text.replace(graph, "").replace("28 tool calls and 7 minutes", "36 tool calls and 6 minutes")
    anchor = "- Never look outside /workspace."
    assert text.count(anchor) == 1
    prompt.write_text(text.replace(anchor, LB07_RULES + anchor))
    _set_sampling(out, temperature=1.0)


LB08_MEMORY = (
    "# MEMORY (survives history compaction)\n"
    "- During the task, older tool results are replaced by a short summary. Exact file text you read "
    "earlier will then be gone. Keep a memory file at /tmp/notes.md (outside /workspace, so it is "
    "never part of the patch).\n"
    "- Writing notes must never cost an extra tool call: chain it onto a run_command you are making "
    "anyway.\n"
    "- Read the region you intend to edit with `echo '== path A-B ==' >> /tmp/notes.md; sed -n "
    "'A,Bp' path | tee -a /tmp/notes.md`, so the exact text is saved as you read it.\n"
    "- When you learn something decisive (target located, edit made, test result), append one line "
    "in the same command: `...; echo 'NOTE: <fact>' >> /tmp/notes.md`.\n"
    "- If you need exact text or a result you can no longer see (only a summary), run "
    "`cat /tmp/notes.md` once before reading any file again. Never repeat a search or read whose "
    "result is in the notes.\n"
)


def lb08_notes(out: Path) -> None:
    """lb04 plus an agent-kept memory file (/tmp/notes.md) written with tee during reads, so the
    exact text to edit survives ADK events compaction (Track W: 32/60 redundant re-reads followed
    compaction). Compaction config belongs to the host; bundles cannot change it or add code
    callbacks (closed registries), so the fix lives in the prompt."""
    lb04_public_7min(out)
    prompt = out / "prompts" / "system.md"
    text = prompt.read_text()
    old = "- Never look outside /workspace. All dependencies"
    assert text.count(old) == 1
    text = text.replace(
        old, "- Never look outside /workspace, except your memory file /tmp/notes.md. All dependencies"
    )
    anchor = "# PROCEDURE"
    assert text.count(anchor) == 1
    prompt.write_text(text.replace(anchor, LB08_MEMORY + "\n" + anchor))


SKILLS_DIR = ROOT / "_skills"

LB09_TOOLS = (
    "# HELPER TOOLS (from the swe-tools skill)\n"
    "- Your FIRST tool call: run_skill_script with skill_name `swe-tools` and file_path "
    "`scripts/install.py`, and no other arguments. It installs three helpers in /tmp/tools. Never "
    "call run_skill_script again; use the helpers only through run_command.\n"
    "- `python /tmp/tools/locate.py \"<exact identifiers, error message words and key nouns from "
    "the issue>\"` ranks the source files most likely to need the change and shows their best "
    "matching lines. Use it as your first search, instead of several greps.\n"
    "- `python /tmp/tools/outline.py <path>` lists the classes and functions of a file with line "
    "ranges; `python /tmp/tools/outline.py <path> <name>` prints that function or class (for "
    "example `Client.send`) with line numbers. Prefer it to reading long files.\n"
    "- `python /tmp/tools/replace.py <path>` with a quoted heredoc (a line `@@OLD`, the exact old "
    "lines, a line `@@NEW`, the new lines) replaces one exact block and refuses edits that would "
    "not compile. Use it for multi-line edits and whenever the old text contains a docstring or "
    "quotes.\n"
)


def lb09_skill_tools(out: Path) -> None:
    """lb04 plus a bundled skill whose install script puts three stdlib helpers in /tmp/tools:
    locate.py (BM25 + exact-identifier ranking; gold file in top-5 for 23/38, top-10 28/38 of the
    screen tasks offline), outline.py (AST outline / one function with line numbers) and
    replace.py (exact unique block replacement with a compile check, avoiding edit_file argument
    mangling). run_skill_script is called once without list arguments (gemma4 parser freeze on
    array args, forum 753973); the helpers then run through run_command."""
    lb04_public_7min(out)
    shutil.copytree(SKILLS_DIR / "swe-tools", out / "skills" / "swe-tools",
                    ignore=shutil.ignore_patterns("__pycache__"))
    agent = out / "agent.yaml"
    text = agent.read_text()
    anchor = "generate_content_config:"
    assert text.count(anchor) == 1
    agent.write_text(text.replace(anchor, "skills:\n  - skills/swe-tools\n" + anchor))
    prompt = out / "prompts" / "system.md"
    text = prompt.read_text()
    anchor = "# CALL PLAN"
    assert text.count(anchor) == 1
    prompt.write_text(text.replace(anchor, LB09_TOOLS + "\n" + anchor))


LB10_PLAN = (
    "# WORKING PLAN (costs no tool calls)\n"
    "- Before your first tool call, write a short plan as plain text in this exact shape:\n"
    "  PLAN\n"
    "  Goal: <one line: what must change and where the hidden tests will look>\n"
    "  1. Locate <symbol or file>\n"
    "  2. Read <region>\n"
    "  3. Edit <file: the change>\n"
    "  4. Verify <test file or python -c check>\n"
    "  5. git status + git diff, submit_patch\n"
    "- Start every later message with one status line: `[step k/5 | call n/28 | edited: yes/no]` "
    "followed by the next action. Count your calls honestly.\n"
    "- When you learn something that changes the plan (the real file, a second place to edit), "
    "rewrite the PLAN block in full in that message, then continue.\n"
    "- If earlier messages seem to be missing (the conversation was shortened), first rewrite the "
    "PLAN with what you already know: files found, edits made, what remains. Do not repeat searches "
    "whose results you have already acted on.\n"
    "- If the status line shows `edited: no` at call 12 or later, your next call is an edit.\n"
)


def lb10_plan(out: Path) -> None:
    """lb04 plus a zero-call text planning layer (harness study arXiv 2609.20804: a to-do plan
    re-shown each turn cut no-edit runs 68.6% -> 27.8%). The plan and a per-message status line are
    plain text, so they spend no tool calls; a re-plan rule covers host-side event compaction.
    lb08 (notes file) and lb09 (skill helpers) were not stacked: both screened below lb04."""
    lb04_public_7min(out)
    prompt = out / "prompts" / "system.md"
    text = prompt.read_text()
    anchor = "# PROCEDURE"
    assert text.count(anchor) == 1
    prompt.write_text(text.replace(anchor, LB10_PLAN + "\n" + anchor))



def lb11_plan_early_edit(out: Path) -> None:
    """lb10_plan with the forced-edit deadline moved from call 12 to call 10 (locate 1-6, edit 7-10).
    lb10 still hit the 7-min cap on 16/39 tasks and left 9 empty patches; an earlier edit leaves more
    calls and time to verify and correct."""
    lb10_plan(out)
    prompt = out / "prompts" / "system.md"
    text = prompt.read_text()
    swaps = [
        ("- Calls 1-8: locate and read. Calls 9-12: edit. You MUST have edited a source file by call 12.",
         "- Calls 1-6: locate and read. Calls 7-10: edit. You MUST have edited a source file by call 10."),
        ("- Calls 13-20: verify and correct.", "- Calls 11-20: verify and correct."),
        ("`edited: no` at call 12 or later", "`edited: no` at call 10 or later"),
    ]
    for old, new in swaps:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    prompt.write_text(text)



LB12_SHORT = (
    "- Keep your reasoning short: a few lines, then act. If you notice you are repeating a line or "
    "re-checking the same fact, stop thinking and make the tool call.\n"
)


def lb12_loop_cap(out: Path) -> None:
    """lb10_plan plus caps on Gemma 4 thinking repetition loops (IDEAS addendum 3): in lb04 32 steps in
    21/39 tasks loop to the 4096 thinking budget (median 58 s each, 28% of generation time; looping
    tasks resolve 4/21 vs 12/18). thinking_budget 4096 -> 1024 and max_output_tokens 8192 -> 3072 bound
    the cost of any one loop; one prompt rule asks for short reasoning. Bundles have no hooks, so the
    loop cannot be detected and stopped mid-generation."""
    lb10_plan(out)
    cfg = out / "configs" / "sampling.yaml"
    text = cfg.read_text()
    for old, new in (("max_output_tokens: 8192", "max_output_tokens: 3072"),
                     ("thinking_budget: 4096", "thinking_budget: 1024")):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    cfg.write_text(text)
    prompt = out / "prompts" / "system.md"
    text = prompt.read_text()
    anchor = "# PROCEDURE"
    assert text.count(anchor) == 1
    prompt.write_text(text.replace(anchor, LB12_SHORT + "\n" + anchor))



LB13_LOCATOR_YAML = """name: code_locator
description: Read-only helper. Give it the issue's requirement and every exact identifier, message and file name from the issue; it finds the source location to change and returns the target, related places, the current code snippet and the likely test file.
model: gemma-4-31b-it-qat-w4a16-ct
instruction: !include ../prompts/locator.md
tools:
  - run_command
  - read_file
generate_content_config: !include ../configs/sampling.yaml
"""

LB13_LOCATOR_PROMPT = """You are a read-only code locator for the repository in /workspace. The request describes an issue. Find the source location that must change so the issue is fixed, then report it. Another agent will make the edit from your report.

# RULES
- At most 6 tool calls. Every call you make is taken from the other agent's budget, so stop as soon as you have found the place.
- Never edit, create or delete files. Never run tests. Never look outside /workspace.
- Search with run_command: combine several fixed-string searches for the issue's exact identifiers, messages and option names into one command, for example `git grep -n -F -e 'name_one' -e 'name_two' -- '*.py' ':!tests' | head -40`. Use `git ls-files | grep -i <word> | head` to find files by name.
- Read only small regions: `sed -n '120,170p' path` (about 50 lines), or read_file for one region.
- For a bug, the target is where the wrong behaviour is produced, not where it is observed. For a feature, the target is the closest existing sibling to copy (a similar option, function or class) and every registration point it needs (__all__, option lists, dispatch tables, imports).
- Note parallel implementations that need the same change: sync and async variants, several backends, re-exports.
- For FastAPI documentation tasks, the target is the executable code under docs_src/.
- Keep your reasoning short. If you notice you are repeating yourself, stop and answer.

# ANSWER
Finish with plain text and no tool call, at most 40 lines, in exactly this shape:
TARGET: <path>:<start>-<end> <function or class> - <one line: what must change>
ALSO: <other path:line places that need the same or a related change, or "none">
SNIPPET:
<the current code of the target region, at most 25 lines, each line prefixed with its line number>
TESTS: <the most likely existing test file for this behaviour, or "unknown">
If you are not sure, give your best two candidates as TARGET and ALSO, best first.
"""

LB13_MAIN = (
    "# LOCATOR TOOL\n"
    "- Your FIRST tool call is `code_locator`. In its request, write the issue's requirement in one or two "
    "sentences followed by every exact identifier, error message, option and file name from the issue.\n"
    "- It searches in a separate context and returns TARGET, ALSO, SNIPPET (current code with line "
    "numbers) and TESTS. Its own searches count against your 28 calls (it uses up to 6).\n"
    "- Trust its report. Do not repeat its searches. Read more only if the SNIPPET is not enough to "
    "write the edit, then edit TARGET (and every ALSO place that needs it).\n"
    "- Call it only once. If its report is empty or clearly wrong, fall back to the Locate procedure "
    "below.\n"
)


def lb13_locator(out: Path) -> None:
    """lb10_plan plus a read-only `code_locator` agent_tool (HARNESS_README best practice 4): searching
    and reading happen in the sub-agent's own context and only a <=40-line report enters the main
    context, which the host compacts at ~14k tokens (32/60 re-reads followed compaction). Sub-agent tool
    calls share the 28-call / 7-min budget (counted in swegemma ToolContext), so it is capped at 6 calls.
    No LoRA yet: this tests the structure alone before training a locator adapter."""
    lb10_plan(out)
    (out / "sub_agents").mkdir(exist_ok=True)
    (out / "sub_agents" / "code_locator.yaml").write_text(LB13_LOCATOR_YAML)
    (out / "prompts" / "locator.md").write_text(LB13_LOCATOR_PROMPT)
    agent = out / "agent.yaml"
    text = agent.read_text()
    anchor = "generate_content_config:"
    assert text.count(anchor) == 1
    tool = ("  - agent_tool:\n      config_path: sub_agents/code_locator.yaml\n"
            "      skip_summarization: true\n")
    agent.write_text(text.replace(anchor, tool + anchor))
    prompt = out / "prompts" / "system.md"
    text = prompt.read_text()
    anchor = "# CALL PLAN"
    assert text.count(anchor) == 1
    text = text.replace(anchor, LB13_MAIN + "\n" + anchor)
    old = "  1. Locate <symbol or file>\n"
    assert text.count(old) == 1
    text = text.replace(old, "  1. code_locator for <key identifiers>\n")
    prompt.write_text(text)


BUILDERS: dict[str, Callable[[Path], None]] = {
    "lb13_locator": lb13_locator,
    "lb12_loop_cap": lb12_loop_cap,
    "lb11_plan_early_edit": lb11_plan_early_edit,
    "lb10_plan": lb10_plan,
    "lb09_skill_tools": lb09_skill_tools,
    "lb08_notes": lb08_notes,
    "lb07_safe_tools": lb07_safe_tools,
    "lb06_12b": lb06_12b,
    "lb05a_think2048": lb05a_think2048,
    "lb05b_audit_rules": lb05b_audit_rules,
    "lb04_public_7min": lb04_public_7min,
    "lb03_thinking_off": lb03_thinking_off,
    "lb03h_think_cap4k": lb03h_think_cap4k,
    "lb01_reserve_fix": lb01_reserve_fix,
    "lb02_no_lora_t60": lb02_no_lora_t60,
}


async def _stub(**_: object) -> str:
    return ""


def validate(sub: Path) -> dict[str, object]:
    files = [p for p in sub.rglob("*") if p.is_file()]
    bad = [str(p.relative_to(sub)) for p in files if p.suffix not in ALLOWED]
    links = [str(p.relative_to(sub)) for p in sub.rglob("*") if p.is_symlink()]
    size = sum(p.stat().st_size for p in files)
    if bad or links or size >= MAX_BYTES:
        raise ValueError(f"invalid submission: bad={bad} symlinks={links} size={size}")
    if (sub / "skills").exists():
        # swebox's own compiler has no skill support; the official adk_submission compiler (used by
        # tools/official_screen.py and the scorer) validates skill bundles instead.
        return {"files": len(files), "bytes": size, "skills": "validated by official harness only"}
    compiled = compile_submission(sub, dict.fromkeys(TOOLS, _stub), ServingTarget())
    return {
        "files": len(files),
        "bytes": size,
        "model": compiled.model,
        "adapters": compiled.adapters,
        "agents": compiled.agent_count,
    }


def build(name: str) -> Path:
    out = ROOT / name
    if out.exists():
        shutil.rmtree(out)
    BUILDERS[name](out)
    info = validate(out)
    # Kaggle requires the upload to be named submission.zip; the version lives in the folder.
    zdir = data_root() / "submissions" / name
    zdir.mkdir(parents=True, exist_ok=True)
    zpath = zdir / "submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out))  # files at the zip root, as the rules require
    print(f"{name}: {info} -> {zpath} ({zpath.stat().st_size / 1e6:.2f} MB)")
    return zpath


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(BUILDERS))
    build(ap.parse_args().name)
