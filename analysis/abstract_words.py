r"""Word count for the MDPI abstract, counting a math span as ONE word.

A naive split overcounts badly: "$[-12.5,\,-5.8]$" becomes four tokens and
"$+1.9$\\,pp" becomes two, which inflates the count by ~10% and makes the
200-word compliance check meaningless. Math spans are collapsed to a single
placeholder first, then ordinary tokenisation is applied.
"""
import re
import sys

src = open("paper/main.tex").read()
m = re.search(r"\\abstract\{%(.*?)\}\s*\\keyword", src, re.S)
if not m:
    sys.exit("abstract not found")
body = m.group(1)
body = re.sub(r"\$[^$]*\$", " MATH ", body)          # each math span -> 1 token
body = re.sub(r"\\[a-zA-Z]+\s*", " ", body)          # drop control words
body = re.sub(r"[{}~\\]", " ", body)
words = [w for w in re.split(r"\s+", body) if re.search(r"[A-Za-z0-9]", w)]
n = len(words)
print(f"abstract words: {n} / 200")
if n > 200:
    print(f"  OVER by {n-200}")
print("text:", " ".join(words))
