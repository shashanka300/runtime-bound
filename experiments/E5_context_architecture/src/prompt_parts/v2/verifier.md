
## Localization
{localization}

## Edit just made
{edit_summary}

## Your phase: VERIFY (at most 6 tool calls)
1. If a relevant test file is known, run only it: `python3 -m pytest <file> -x -q`. Otherwise write a 5-15 line check to /tmp/check.py with `run_command` (cat > /tmp/check.py << 'EOF' ... EOF) that exercises the fixed behaviour, and run it.
2. Run `git status --short`; delete scratch files the edit left inside /workspace (never repository files).
3. If the check passes and the tests do not fail because of the change: call `submit_patch`, then reply DONE.
4. Otherwise do NOT submit. Reply with this note and nothing else (at most 120 words):
VERIFICATION FAILED
observed: <what still fails, with the key error line>
suggestion: <what the next edit should change>
