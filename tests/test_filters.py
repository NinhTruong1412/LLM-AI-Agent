import pytest

from hybrid_retrieval.filters import build_mask, matches


@pytest.mark.parametrize(
    "flt,expected",
    [
        ({"source": "arxiv"}, True),
        ({"source": "pubmed"}, False),
        ({"year": {"$gte": 2020}}, True),
        ({"year": {"$lt": 2020}}, False),
        ({"source": {"$in": ["arxiv", "book"]}}, True),
        ({"source": {"$nin": ["arxiv"]}}, False),
        ({"missing": {"$exists": False}}, True),
        ({"missing": "x"}, False),
        ({"$or": [{"year": 1999}, {"source": "arxiv"}]}, True),
        ({"$and": [{"year": 2020}, {"source": "book"}]}, False),
        ({"year": {"$gt": "not-a-number"}}, False),
    ],
)
def test_matches(flt, expected):
    assert matches({"year": 2020, "source": "arxiv"}, flt) is expected


def test_unknown_operator_raises():
    with pytest.raises(ValueError):
        matches({"a": 1}, {"a": {"$regex": "x"}})


def test_build_mask(docs):
    assert build_mask(docs, None) is None
    mask = build_mask(docs, {"source": "pubmed"})
    assert [d.id for d, m in zip(docs, mask, strict=True) if m] == ["d1", "d2", "d5"]
