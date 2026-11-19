"""Tokenisation shared by the sparse retriever. Deliberately simple and fast."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

# A compact English stopword list; avoids pulling in NLTK.
STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have he her
    his i if in into is it its may might more most no not of on or our she should so such than
    that the their them then there these they this those to was we were what when where which
    while who will with would you your""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
