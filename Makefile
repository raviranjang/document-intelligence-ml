UV ?= uv

.PHONY: build format lint sync test typecheck validate

sync:
	$(UV) sync --locked --dev

format:
	$(UV) run --locked ruff format .

lint:
	$(UV) run --locked ruff check .

typecheck:
	$(UV) run --locked mypy

test:
	$(UV) run --locked pytest

build:
	$(UV) build

validate:
	$(UV) lock --check
	$(UV) run --locked ruff format --check .
	$(UV) run --locked ruff check .
	$(UV) run --locked mypy
	$(UV) run --locked pytest
	$(UV) build
