import pytest

from hybrid_retrieval.types import Document

CORPUS = [
    (
        "d1",
        "Aspirin reduces the risk of heart attack in adults.",
        {"year": 2019, "source": "pubmed"},
    ),
    (
        "d2",
        "Statins lower cholesterol and cardiovascular events.",
        {"year": 2021, "source": "pubmed"},
    ),
    (
        "d3",
        "Transformers use self-attention for sequence modelling.",
        {"year": 2017, "source": "arxiv"},
    ),
    (
        "d4",
        "Retrieval augmented generation grounds language models in documents.",
        {"year": 2020, "source": "arxiv"},
    ),
    (
        "d5",
        "Myocardial infarction outcomes improve with early aspirin therapy.",
        {"year": 2022, "source": "pubmed"},
    ),
    (
        "d6",
        "BM25 is a strong lexical baseline for document retrieval.",
        {"year": 2009, "source": "book"},
    ),
    (
        "d7",
        "Dense retrievers embed queries and documents in a shared vector space.",
        {"year": 2020, "source": "arxiv"},
    ),
    (
        "d8",
        "Cross-encoders rerank candidate documents with full attention.",
        {"year": 2019, "source": "arxiv"},
    ),
]


@pytest.fixture
def docs():
    return [Document(id=i, text=t, metadata=m) for i, t, m in CORPUS]
