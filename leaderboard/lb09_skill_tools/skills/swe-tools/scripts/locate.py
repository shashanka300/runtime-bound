"""Rank source files for an issue: BM25 over identifier sub-tokens + exact identifier hits.

    python /tmp/tools/locate.py "<identifiers, error words, key nouns>" [--tests]

Prints the top 10 source files (tests excluded unless --tests) with their best matching lines.
Stdlib only; works on Python 3.6+.
"""
import math
import os
import re
import subprocess
import sys
from collections import Counter

ROOT = "/workspace" if os.path.isdir("/workspace") else os.getcwd()
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STOP = set(
    "the a an and or of to in on for is are be with this that it as by not no if when from at "
    "should would can could does do use using used get set none true false self cls def class "
    "return import raise error issue bug fix".split()
)
TEST_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$")
NOISE_RE = re.compile(r"(^|/)(examples?|benchmarks?|docs|_unicode_data|vendor|_vendor)(/|$)|(^|/)(__main__|setup|noxfile)\.py$")


def subtokens(word):
    out = []
    for part in re.split(r"_+", word):
        out += re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", part)
    return [t.lower() for t in out if len(t) > 1]


def files(include_tests):
    try:
        names = subprocess.check_output(
            ["git", "ls-files"], cwd=ROOT, universal_newlines=True
        ).splitlines()
    except Exception:
        names = []
        for dp, _, fs in os.walk(ROOT):
            names += [os.path.relpath(os.path.join(dp, f), ROOT) for f in fs]
    keep = []
    for n in names:
        if not n.endswith((".py", ".pyi", ".rst", ".md", ".txt", ".cfg", ".toml")):
            continue
        if not n.endswith((".py", ".pyi")) and not n.startswith("docs_src/"):
            continue
        if not include_tests and TEST_RE.search(n):
            continue
        if NOISE_RE.search(n) and not n.startswith("docs_src/"):
            continue
        keep.append(n)
    return keep


def main():
    args = [a for a in sys.argv[1:] if a != "--tests"]
    if not args:
        print(__doc__)
        return
    query = " ".join(args)
    exact = [w for w in IDENT.findall(query) if (("_" in w) or any(c.isupper() for c in w[1:]) or len(w) > 3) and w.lower() not in STOP]
    q_terms = []
    for w in IDENT.findall(query):
        if w.lower() in STOP:
            continue
        q_terms += [t for t in subtokens(w) if t not in STOP]
    q_terms = list(dict.fromkeys(q_terms))
    phrases = []
    for a, b in re.findall(r"'([^']{6,})'|\"([^\"]{6,})\"", query):
        if (a or b).strip():
            phrases.append((a or b).strip())

    docs = {}
    for n in files("--tests" in sys.argv):
        try:
            with open(os.path.join(ROOT, n), encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except Exception:
            continue
        if len(text) > 2000000:
            continue
        docs[n] = text
    if not docs:
        print("no files found under " + ROOT)
        return

    tf = {}
    df = Counter()
    lens = {}
    for n, text in docs.items():
        toks = []
        for w in IDENT.findall(text):
            toks += subtokens(w)
        toks += subtokens(n.replace("/", "_").replace(".py", ""))
        c = Counter(toks)
        tf[n] = c
        lens[n] = len(toks) or 1
        for t in set(q_terms):
            if c.get(t):
                df[t] += 1
    N = len(docs)
    avg = sum(lens.values()) / float(N)
    scores = {}
    for n in docs:
        s = 0.0
        c = tf[n]
        for t in q_terms:
            f = c.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * lens[n] / avg))
        text = docs[n]
        for w in exact:
            if re.search(r"\b(def|class)\s+" + re.escape(w) + r"\b", text):
                s += 12.0
            elif re.search(r"\b" + re.escape(w) + r"\b", text):
                s += 3.0
        for p in phrases:
            if p in text:
                s += 8.0
        base = os.path.basename(n).replace(".py", "").lower()
        if base in [t.lower() for t in exact] or base in q_terms:
            s += 4.0
        scores[n] = s
    top = sorted(scores, key=lambda k: -scores[k])[:10]
    print("query terms: " + " ".join(q_terms[:20]))
    for n in top:
        if scores[n] <= 0:
            break
        print("%6.1f  %s" % (scores[n], n))
        hits = []
        for i, line in enumerate(docs[n].splitlines(), 1):
            ls = line.strip()
            score = 0
            for w in exact:
                if re.search(r"\b" + re.escape(w) + r"\b", line):
                    score += 3 if re.match(r"(async\s+)?(def|class)\s", ls) else 2
            for p in phrases:
                if p in line:
                    score += 3
            if score:
                hits.append((score, i, ls[:110]))
        for score, i, ls in sorted(hits, key=lambda h: (-h[0], h[1]))[:3]:
            print("        %d: %s" % (i, ls))


if __name__ == "__main__":
    main()
