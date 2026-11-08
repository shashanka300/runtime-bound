
## Your phase: LOCATE (read-only)
Find where the issue must be fixed. Do not edit anything and do not create files.
1. Pull names, error messages, functions and classes out of the issue; locate them with `grep -rn` (limit output with `| head -30`) or `search_similar_code` / `get_code_neighbors`.
2. Read only the relevant functions (small line ranges).
3. Stop as soon as you know the file(s), the function(s) and the root cause. Use at most about 12 tool calls.

End with this note and nothing else (at most 250 words):
LOCALIZATION
files: <path:line-range> for each place that must change
root cause: <one or two sentences>
fix plan: <the minimal change, in words>
relevant tests: <test file(s) that exercise this code, if found>
