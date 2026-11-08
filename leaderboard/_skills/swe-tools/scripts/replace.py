"""Replace one exact block in a file. Old and new text come from stdin:

    python /tmp/tools/replace.py <path> <<'EOF'
    @@OLD
    <exact old lines>
    @@NEW
    <new lines>
    EOF

The old block must occur exactly once (whitespace-exact; a trailing-whitespace-insensitive match is
used as a fallback). A .py file must still compile, otherwise nothing is written.
"""
import os
import py_compile
import re
import sys
import tempfile

ROOT = "/workspace" if os.path.isdir("/workspace") else os.getcwd()


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    data = sys.stdin.read()
    m = re.match(r"\A\s*@@OLD[ \t]*\n(.*?)\n@@NEW[ \t]*\n(.*)\Z", data, re.S)
    if not m:
        print("stdin must be: @@OLD line, old text, @@NEW line, new text")
        sys.exit(1)
    old, new = m.group(1), m.group(2)
    if new.endswith("\n"):
        new = new[:-1]
    with open(full, encoding="utf-8") as fh:
        src = fh.read()
    count = src.count(old)
    if count != 1:
        strip = lambda s: "\n".join(l.rstrip() for l in s.split("\n"))
        if count == 0 and strip(src).count(strip(old)) == 1:
            src, old = strip(src), strip(old)
        else:
            print("old block found %d times in %s; nothing written. Read the exact lines and retry "
                  "with a unique block." % (count, path))
            sys.exit(1)
    out = src.replace(old, new, 1)
    if full.endswith(".py"):
        fd, tmp = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(out)
        try:
            py_compile.compile(tmp, doraise=True)
        except py_compile.PyCompileError as e:
            print("result would not compile; nothing written:\n" + str(e)[-600:])
            sys.exit(1)
        finally:
            os.unlink(tmp)
    with open(full, "w", encoding="utf-8") as fh:
        fh.write(out)
    start = src[: src.index(old)].count("\n") + 1
    print("replaced %d line(s) with %d line(s) at %s:%d" % (old.count("\n") + 1, new.count("\n") + 1, path, start))


if __name__ == "__main__":
    main()
