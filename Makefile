.PHONY: serve seed index eval eval-all smoke plots lint format test clean

serve:
	uvicorn app.main:app --reload --port 8000

seed:
	python -m scripts.seed_corpus

index:
	python -m scripts.build_index_from_corpus

eval:
	python -m app.eval.run_eval --mode hybrid_rrf --ks 1,3,5,10

eval-all:
	python -m app.eval.run_eval --mode vector --ks 1,3,5,10
	python -m app.eval.run_eval --mode bm25 --ks 1,3,5,10
	python -m app.eval.run_eval --mode hybrid_rrf --ks 1,3,5,10

smoke:
	python -m app.eval.smoke_eval

plots:
	T1_EVAL_MODE=hybrid_rrf python -m app.eval.report_plots

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

test:
	pytest -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
