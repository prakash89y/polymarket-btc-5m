.PHONY: install install-dev test cov lint fmt type check doctor docker clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

cov:
	pytest --cov=pmbtc --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

type:
	mypy

# What CI runs, and what should pass before any module is called done.
check: lint type test

doctor:
	pmbtc doctor

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
