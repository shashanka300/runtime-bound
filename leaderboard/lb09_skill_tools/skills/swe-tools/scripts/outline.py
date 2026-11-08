"""Outline a Python file, or print one class/function with line numbers.

    python /tmp/tools/outline.py <path>            # classes/functions with line ranges
    python /tmp/tools/outline.py <path> <name>     # source of that def/class (Class.method ok)
"""
import ast
import os
import sys

ROOT = "/workspace" if os.path.isdir("/workspace") else os.getcwd()


def ranges(tree, nlines):
    out = []

    def walk(body, prefix, parent_end):
        nodes = [n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        for i, n in enumerate(nodes):
            start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
            end = getattr(n, "end_lineno", None)
            if end is None:
                end = (nodes[i + 1].lineno - 1) if i + 1 < len(nodes) else parent_end
            name = prefix + n.name
            kind = "class" if isinstance(n, ast.ClassDef) else "def"
            out.append((name, kind, start, end))
            if isinstance(n, ast.ClassDef):
                walk(n.body, name + ".", end)

    walk(tree.body, "", nlines)
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    path = sys.argv[1]
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full, encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    lines = src.splitlines()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print("cannot parse %s: %s" % (path, e))
        return
    items = ranges(tree, len(lines))
    if len(sys.argv) == 2:
        print("%s: %d lines" % (path, len(lines)))
        for name, kind, a, b in items:
            depth = name.count(".")
            if depth > 1:
                continue
            print("%s%s %s  %d-%d" % ("    " * depth, kind, name.split(".")[-1], a, b))
        return
    want = sys.argv[2]
    found = [it for it in items if it[0] == want or it[0].endswith("." + want)]
    if not found:
        print("no def/class named %r in %s" % (want, path))
        return
    for name, kind, a, b in found[:3]:
        if b - a > 160:
            print("# %s %s %d-%d (long; showing first 160 lines)" % (kind, name, a, b))
            b = a + 159
        else:
            print("# %s %s %d-%d" % (kind, name, a, b))
        for i in range(a, b + 1):
            print("%5d  %s" % (i, lines[i - 1]))


if __name__ == "__main__":
    main()
