import polars as pl
from src import taxonomy as tx

GOLD = """--- a/pkg/core.py
+++ b/pkg/core.py
@@ -40,6 +40,7 @@ def f():
     a = 1
+    b = 2
"""


def _patch(path: str, start: int) -> str:
    return (
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        f"@@ -{start},3 +{start},4 @@\n x\n+y\n"
    )


def test_localization_levels() -> None:
    assert tx.localization("", GOLD) == "no_patch"
    assert tx.localization(_patch("pkg/other.py", 40), GOLD) == "mislocalized_file"
    assert tx.localization(_patch("pkg/core.py", 200), GOLD) == "mislocalized_region"
    assert tx.localization(_patch("pkg/core.py", 44), GOLD) == "right_region"  # within slack
    assert tx.localization(_patch("tests/test_core.py", 40), GOLD) == "no_patch"  # tests only


def test_scratch_files_are_not_localization() -> None:
    scratch = (
        "diff --git a/tmp/repro.py b/tmp/repro.py\nnew file mode 100644\n"
        "--- /dev/null\n+++ b/tmp/repro.py\n@@ -0,0 +1,2 @@\n+a\n+b\n"
    )
    assert tx.scratch_files(scratch, GOLD) == {"tmp/repro.py"}
    assert tx.localization(scratch, GOLD) == "no_patch"
    assert tx.localization(scratch + _patch("pkg/core.py", 41), GOLD) == "right_region"


def test_primary_precedence() -> None:
    ctx = {"agent_error": "litellm.ContextWindowExceededError: ...", "end_reason": "error"}
    assert tx.primary_label(ctx, "right_region") == "context_exhausted"
    assert tx.primary_label({"end_reason": "time"}, "right_region") == "time_exhausted"
    assert tx.primary_label({"end_reason": "submitted"}, "no_patch") == "gave_up_no_patch"
    assert (
        tx.primary_label({"end_reason": "submitted", "patch_applied": False}, "right_region")
        == "patch_apply_failed"
    )
    assert (
        tx.primary_label({"end_reason": "submitted", "patch_applied": True}, "right_region")
        == "wrong_edit"
    )


def test_behaviour_markers() -> None:
    base = {
        "tag": "t",
        "instance_id": "i_1",
        "seed": 0,
        "kind": "tool",
        "start_line": None,
        "end_line": None,
        "filepath": None,
    }
    rows = [
        {**base, "step": 1, "tool": "run_command", "args": '{"command": "cat pkg/core.py"}'},
        {**base, "step": 2, "tool": "run_command", "args": '{"command": "cat pkg/core.py"}'},
        {
            **base,
            "step": 3,
            "tool": "edit_file",
            "args": '{"filepath": "x.py", "old_string": "A", "new_string": "B"}',
        },
        {
            **base,
            "step": 4,
            "tool": "edit_file",
            "args": '{"filepath": "x.py", "old_string": "B", "new_string": "A"}',
        },
    ]
    b = tx.behaviour(pl.DataFrame(rows), {"i_1": {"pkg/core.py"}}).row(0, named=True)
    assert b["repeated_actions"] == 1 and b["reverted_edits"] == 1 and b["viewed_gold_file"]
