
## Your phase: LOCATE (read-only, at most 8 tool calls)
Find where the issue must be fixed. Do not edit anything and do not create files.
Budget: at most 8 tool calls in total. After your 8th call you MUST write the note below, even if you are not certain; the next phase can read more.
1. Take the most specific names from the issue (functions, classes, error messages) and find them in one call, e.g. `grep -rn "name" --include=*.py src pkg | head -20`, or `search_similar_code` / `get_code_neighbors` with a symbol name.
2. Read only the function that must change (a small line range).
3. Write the note as soon as you know the file and function.

End with this note and nothing else (at most 200 words):
LOCALIZATION
files: <path:line-range> for each place that must change
root cause: <one or two sentences>
fix plan: <the minimal change, in words>
relevant tests: <test file(s) that exercise this code, if seen; otherwise none>
