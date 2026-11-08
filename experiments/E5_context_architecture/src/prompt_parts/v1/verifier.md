
## Localization
{localization}

## Reproduction
{repro}

## Edit just made
{edit_summary}

## Your phase: VERIFY
1. Run the reproduction command and, if a relevant test file is known, run only that file with `python3 -m pytest <file> -x -q`.
2. Run `git status --short` and delete any scratch files the edit left inside /workspace (never delete repository files).
3. If the reproduction now shows the expected result and the targeted tests do not fail because of the change: call `submit_patch`, then reply DONE.
4. Otherwise do NOT submit. Reply with this note and nothing else (at most 150 words):
VERIFICATION FAILED
observed: <what still fails, with the key error line>
suggestion: <what the next edit should change>
