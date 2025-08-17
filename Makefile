# GraphVerse Makefile
# Simple commands for development and experiments

.PHONY: help install install-dev test lint format clean demo demo-multi experiments

help:  ## Show this help message
	@echo "GraphVerse - Rule Learning in Small LLMs"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run tests
	cd tests && python run_tests.py

lint:  ## Run code linting
	flake8 graphverse/
	black --check graphverse/

format:  ## Format code with black
	black graphverse/
	black scripts/

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf experiments/
	rm -rf multi_model_experiments_*/

demo:  ## Run quick demo (single model)
	python run_analysis.py --mode single --config small

demo-multi:  ## Run multi-model comparison demo
	python run_analysis.py --mode multi --config multi

experiments: ## Run full experiments (longer)
	python run_analysis.py --mode single --config default

# Development targets
dev-setup: install-dev  ## Complete development setup
	@echo "Development environment ready!"

check: test lint  ## Run all checks

# Documentation
docs:  ## Generate documentation (placeholder)
	@echo "Documentation generation not implemented yet"

# Release targets
build:  ## Build package
	python setup.py sdist bdist_wheel

upload-test:  ## Upload to test PyPI
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload:  ## Upload to PyPI
	twine upload dist/*

# Quick start shortcuts  
quick: demo  ## Alias for demo

start: demo  ## Alias for demo

run: demo  ## Alias for demo