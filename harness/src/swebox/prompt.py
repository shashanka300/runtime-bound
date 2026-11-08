"""Initial user prompt (spec 5.2) and continuation nudges (spec 5.3).

Section headings, limits and nudge texts are copied from the spec. The wording of the
"Standard Instructions 0-5" is not published; ours follows the spec's description of them
(work under /workspace, follow conventions, targeted checks, scratch files in /tmp, call
submit_patch, then reply). This is a documented fidelity gap (README: harness fidelity).
"""

from __future__ import annotations

from swebox.tasks import Task
from swebox.tools import Budget

NUDGE_TRUNCATED_TOOL_CALL = (
    "Your previous response reached the token limit before the tool call finished closing "
    "(<|tool_call|> was cut off). Do NOT repeat your prior reasoning in thought—emit your "
    "next tool call immediately, and if calling edit_file or write_file, split the change into "
    "smaller incremental edits."
)
NUDGE_MAX_TOKENS = (
    "Your previous response reached the token limit while thinking before a tool call was "
    "completed. Do NOT repeat your analysis in thought—keep reasoning under a few sentences "
    "and emit your next tool call immediately, or call submit_patch when you have completed and "
    "verified your changes."
)
NUDGE_CONTINUE = (
    "Please continue your work using the available tools, or call submit_patch when you have "
    "completed and verified your changes."
)

STANDARD_INSTRUCTIONS = """\
## Instructions
0. Work only inside /workspace; the repository is already checked out and installed there.
1. Read the relevant code and follow the repository's existing conventions and style.
2. Make the minimal source change that resolves the problem statement.
3. Verify with targeted checks (e.g. inline `python3 -c "..."` assertions or a specific test \
file); avoid running the full test suite.
4. Put any scratch or reproduction files in /tmp, never in /workspace.
5. When done, call `submit_patch`, then reply with a short summary of the change."""

CODE_INTEL = """\
## Code Intelligence Tools
This repository has pre-built code graph and embedding data. Use these tools for fast, \
targeted navigation:
- `search_similar_code(query)`: Find semantically similar functions/classes by keyword.
- `get_code_neighbors(node)`: Find callers, callees, and definitions related to a symbol.
- `get_code_subgraph(nodes)`: Get the induced subgraph for a set of symbols."""


def build_agent_prompt(task: Task, budget: Budget, workspace_listing: str, code_intel: bool) -> str:
    parts = [
        f"You are evaluating a software engineering task for repository {task.repo}.\n\n"
        f"Problem Statement:\n{task.problem_statement}"
    ]
    if task.hints_text.strip():
        parts.append(f"## Hints:\n{task.hints_text}")
    lines = [f"- Time allowance: {budget.time_minutes:.1f} minutes"]
    if budget.tool_calls is not None:
        lines.append(f"- Tool calls allowance: {budget.tool_calls} calls")
    lines.append(f"- Max loop iterations: {budget.turns} turns")
    parts.append(
        "## Task Budget (Session terminates when any budget is exhausted)\n" + "\n".join(lines)
    )
    parts.append(
        "## Execution Environment Rules\n"
        f"- Single command timeout: {budget.command_timeout_seconds} seconds (commands exceeding "
        "this fail without ending the session)\n"
        f"- Command output limit: {budget.max_stdout_chars} characters\n"
        f"- File view limit: {budget.max_file_lines} lines per read_file call\n"
        f"- File character limit: {budget.max_file_chars} characters per read_file call\n"
        "- Environment is offline (no network/PyPI access). All repository and test dependencies "
        "are ALREADY pre-installed. Do NOT attempt to run pip install or download packages."
    )
    parts.append(STANDARD_INSTRUCTIONS)
    if code_intel:
        parts.append(CODE_INTEL)
    parts.append(f"## Workspace Layout\n```\n{workspace_listing}\n```")
    return "\n\n".join(parts)


LISTING_CMD = (
    "find . -maxdepth 3 -not -path './.git*' -not -name '__pycache__' -not -name '*.pyc' "
    "| head -150"
)
