---
name: swe-tools
description: Installs three fast repository helpers (locate, outline, replace) into /tmp/tools. Run scripts/install.py once, with no arguments, at the start of the task.
---
Run `scripts/install.py` once with run_skill_script and no arguments. Afterwards call the helpers
only through run_command:

- `python /tmp/tools/locate.py "<identifiers, error words, key nouns from the issue>"`
  ranks source files and the best matching lines.
- `python /tmp/tools/outline.py <path>` lists classes and functions with line ranges.
  `python /tmp/tools/outline.py <path> <name>` prints that class or function with line numbers.
- `python /tmp/tools/replace.py <path>` replaces one exact block, read from stdin:
  a line `@@OLD`, the exact old text, a line `@@NEW`, the new text.
