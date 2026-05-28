"""Verify core modules import without error (no model/index loading)."""


def test_import_main():
    from app.main import app

    assert app.title == "T1-RAG-REG"


def test_import_retriever():
    from app.rag.retriever import retrieve

    assert callable(retrieve)


def test_import_bm25():
    from app.rag.bm25 import BM25Index

    assert BM25Index is not None


def test_import_calibration():
    from app.rag.calibration import calibrate_best, explain_calibration

    assert callable(calibrate_best)
    bands = explain_calibration()
    assert "vector" in bands
    assert "bm25" in bands
    assert "hybrid_rrf" in bands


def test_import_query_rewrite():
    from app.rag.query_rewrite import rewrite_query

    result = rewrite_query("What is KYC?")
    assert isinstance(result, list)
    assert len(result) >= 1
